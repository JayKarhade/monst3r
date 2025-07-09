import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
from tqdm import tqdm
import rerun as rr
import tempfile
import glob
import shutil
from copy import deepcopy

# Import from sea_raft package
from sea_raft.raft import RAFT
from sea_raft.utils.flow_viz import flow_to_image
from sea_raft.utils.utils import load_ckpt
from sea_raft.parser import parse_args as sea_raft_parse_args

from dust3r.model import AsymmetricCroCo3DStereo  # Assuming Monst3R uses the same model class
from dust3r.image_pairs import make_pairs
from dust3r.utils.geometry import geotrf
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Import dataset modules
from anymap.datasets.multi_view_motion.vkitti2_motion import VKITTI2MultiView_Motion
from anymap.datasets.multi_view_motion.stereo4d_motion import Stereo4DMultiView_Motion
from anymap.datasets.multi_view_motion.pd4d_motion import PD4DMultiView_Motion
from anymap.datasets.multi_view_motion.tapvid3d import TAPVID3DMultiView_Motion
# Import visualization utilities
from anymap.utils.viz import script_add_rerun_args
from anymap.utils.image import rgb
from anymap.utils.misc import seed_everything


def evaluate_scene_flow_metrics(sf_pred, sf_gt):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float32).mean()
    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float32).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float32).mean()

    print(f"EPE3D: {EPE3D:.4f} acc3d_strict: {acc3d_strict:.4f}, acc3d_relax: {acc3d_relax:.4f}, outlier: {outlier:.4f}")


def normalize_multiple_pointclouds(pts_list, valid_masks):
    """
    Normalize multiple pointclouds using average distance to origin.
    
    Args:
        pts_list: List of point clouds, each with shape HxWx3
        valid_masks: List of masks indicating valid points
    
    Returns:
        List of normalized point clouds, normalization factor
    """
    # Replace invalid points with zeros and count valid points
    nan_pts_list = []
    nnz_list = []
    
    for i, pts in enumerate(pts_list):
        mask = valid_masks[i]
        pts_copy = pts.copy()
        pts_copy[~mask] = 0
        nan_pts_list.append(pts_copy)
        nnz_list.append(np.sum(mask))
    
    # Flatten and concatenate all points
    all_pts = np.concatenate([pts.reshape(-1, 3) for pts in nan_pts_list], axis=0)
    
    # Compute distance to origin
    all_dis = np.linalg.norm(all_pts, axis=-1)
    
    # Compute normalization factor (average distance)
    norm_factor = np.sum(all_dis) / (sum(nnz_list) + 1e-8)
    norm_factor = max(norm_factor, 1e-8)  # Prevent division by zero
    
    # Normalize each point cloud
    res = [pts / norm_factor for pts in pts_list]
    
    return res, norm_factor

def convert_sceneflow_ego_to_allo(ego_sf, pts3d, camera_pose):
    """
    Convert ego-centric scene flow to allo-centric using the camera pose.
    
    Args:
        ego_sf: Ego-centric scene flow
        pts3d: 3D points
        pose1: Camera pose matrix
    
    Returns:
        Allo-centric scene flow
    """
    pts3d_ego = pts3d + ego_sf
    pts3d_allo = pts3d_ego @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    allo_sf = pts3d_allo - pts3d
    return allo_sf

def predict_optical_flow(model, args, image1_path, image2_path, device="cuda"):
    """
    Predict optical flow between two images using RAFT

    Args:
        model: RAFT model
        args: arguments for RAFT
        image1_path: first image path
        image2_path: second image path
        device: device to run the model on

    Returns:
        flow: predicted optical flow (numpy array) # (H, W, 2)
        info: additional information from RAFT (numpy array) # (H, W, 4)
    """
    # Load images for RAFT
    image1_rgb = cv2.imread(image1_path)
    image1_rgb = cv2.resize(image1_rgb, (args.img_width, args.img_height))
    image1_rgb = cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.imread(image2_path)
    image2_rgb = cv2.resize(image2_rgb, (args.img_width, args.img_height))
    image2_rgb = cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor format for RAFT - # (B, 3, H, W)
    image1_tensor = torch.from_numpy(image1_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) 
    image2_tensor = torch.from_numpy(image2_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Calculate flow
    with torch.no_grad():
        # Preprocess images
        img1 = F.interpolate(image1_tensor, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2_tensor, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
        
        # Forward pass through the RAFT model
        output = model(img1, img2, iters=args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
            
        # Convert to numpy format
        flow_final = flow_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
        info_final = info_final.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return flow_final, info_final


def prepare_monst3r_input(img_paths, size, device="cuda"):
    """
    Prepare input views for Monst3R from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        size (int): Target image size.
        device (str): Computation device.

    Returns:
        list: A list of view dictionaries.
    """
    # Load and preprocess images
    imgs = load_images(img_paths, size=size, verbose=False)
    pairs = make_pairs(imgs, scene_graph="oneref_mid", prefilter=None, symmetrize=False)

    return pairs


def predict_3d_points_monst3r(model, images_paths, size, device="cuda"):
    """
    Predict 3D points from images using Monst3R

    Args:
        model: Monst3R model
        images_paths: list of image paths
        size: image size for processing
        device: device to run the model on

    Returns:
        pointmaps: 3D point maps (numpy array)
        depth_conf: depth confidence maps
    """
    # Prepare input views
    pairs = prepare_monst3r_input(images_paths, size, device)
    
    # Run inference with Monst3R
    with torch.no_grad():
        output = inference(pairs, model, device, verbose=False)

    mode = GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    lr = 0.01

    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    poses = to_numpy(scene.get_im_poses())

    identity_pointmaps = to_numpy(scene.depth_to_pts3d_identity())

    return identity_pointmaps, confs, poses

    # view1, view2, pred1, pred2 = output['view1'], output['view2'], output['pred1'], output['pred2']
    # pts1 = pred1['pts3d']
    # pts2 = pred2['pts3d_in_other_view']

    # pointmaps = torch.stack([pts1[0], pts2[0]], dim=0).detach().cpu().numpy()

    # depth_conf1 = pred1['conf']
    # depth_conf2 = pred2['conf']
    # depth_conf = torch.stack([depth_conf1[0], depth_conf2[0]], dim=0).detach().cpu().numpy()

    # return pointmaps, depth_conf

def calculate_scene_flow(optical_flow, pointmap1, pointmap2):
    """
    Scene flow calculation from optical flow and pointmaps.
    
    Parameters:
        optical_flow: numpy.ndarray
            Optical flow with shape (H, W, 2), where the last dimension contains (dx, dy)
        pointmap1: numpy.ndarray
            First pointmap with shape (H, W, 3)
        pointmap2: numpy.ndarray
            Second pointmap with shape (H, W, 3)
        
    Returns:
        scene_flow: numpy.ndarray
            Scene flow with shape (H, W, 3)
        valid_mask: numpy.ndarray
            Boolean mask indicating valid flow points
    """
    H, W = optical_flow.shape[:2]

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Calculate target coordinates based on optical flow
    x_target = x_coords + optical_flow[:, :, 0]
    y_target = y_coords + optical_flow[:, :, 1]
    
    # Create a mask for valid flow (inside FOV)
    valid_mask = (x_target >= 0) & (x_target < W) & (y_target >= 0) & (y_target < H)
    
    # Initialize scene flow array with zeros
    scene_flow = np.zeros_like(pointmap1)
    
    # Round to nearest integer and clip to image boundaries for calculation
    x_target_clipped = np.clip(np.round(x_target).astype(np.int32), 0, W-1)
    y_target_clipped = np.clip(np.round(y_target).astype(np.int32), 0, H-1)
    
    # Get the 3D coordinates from the second pointmap at the target locations
    points2 = pointmap2[y_target_clipped, x_target_clipped]
    
    # Calculate scene flow as the difference between 3D points, but only for valid flow points
    scene_flow[valid_mask] = points2[valid_mask] - pointmap1[valid_mask]
    
    return scene_flow, valid_mask


def setup_optical_flow_model(raft_args, ckpt_path, device):
    """
    Set up the RAFT model for optical flow prediction
    
    Args:
        raft_args: arguments for RAFT
        ckpt_path: path to the checkpoint
        device: device to load the model on
        
    Returns:
        model: initialized RAFT model
    """
    model = RAFT(raft_args)
    load_ckpt(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model


def setup_monst3r_model(model_path, device):
    """
    Set up the Monst3R model for depth and pose prediction
    
    Args:
        model_path: path to Monst3R model checkpoint
        device: device to load the model on
        
    Returns:
        model: initialized Monst3R model
    """
    # Add the checkpoint path to dust3r
    # add_path_to_dust3r(model_path)
    
    # Load the model (assuming Monst3R uses ARCroco3DStereo as the base model class)
    model = AsymmetricCroCo3DStereo.from_pretrained('Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt').to(device)
    model.eval()
    
    return model


def log_scene_flow_to_rerun(base_name, view1, view2, pts3d, pred_pts3d, pointmap1, pointmap2, gt_scene_flow, pred_scene_flow, valid_mask, rgb_image1, rgb_image2, sample_ratio=1.0):
    """
    Log ground truth and predicted scene flow to rerun visualization
    
    Args:
        base_name: base name for the log
        view1: first view data dict containing camera parameters
        view2: second view data dict containing camera parameters
        pts3d: 3D points
        pred_pts3d: predicted 3D points
        gt_scene_flow: ground truth scene flow
        pred_scene_flow: predicted scene flow
        valid_mask: mask for valid points
        rgb_image1: RGB image of the first view
        sample_ratio: ratio of points to visualize
    """
    # valid_mask = np.ones_like(valid_mask)
    # Filter points using the valid mask
    masked_pts = pts3d[valid_mask]
    pred_masked_pts = pred_pts3d[valid_mask]
    masked_gt_flow = gt_scene_flow[valid_mask]
    masked_pred_flow = pred_scene_flow[valid_mask]
    masked_colors = rgb_image1[valid_mask]

    # import pdb; pdb.set_trace()
    pointmap1 = pointmap1[valid_mask]
    pointmap2 = pointmap2[valid_mask]
    
    # Sample a subset of points if needed
    if sample_ratio < 1.0 and masked_pts.shape[0] > 0:
        num_points = masked_pts.shape[0]
        sample_size = max(1, int(num_points * sample_ratio))
        indices = np.random.choice(num_points, sample_size, replace=False)
        
        masked_pts = masked_pts[indices]
        pred_masked_pts = pred_masked_pts[indices]
        masked_gt_flow = masked_gt_flow[indices]
        masked_pred_flow = masked_pred_flow[indices]
        masked_colors = masked_colors[indices]
    
    # Log camera info for view 1
    height, width = rgb_image1.shape[0], rgb_image1.shape[1]
    pose1 = view1["camera_pose"]
    intrinsics1 = view1["camera_intrinsics"]
    
    # Log camera transform
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose1[:3, 3],
            mat3x3=pose1[:3, :3],
            from_parent=False,
        ),
    )
    
    # Log camera pinhole model
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics1,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    
    # Log RGB image
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(rgb_image1),
    )

    rr.log(
        f"{base_name}/rgb2",
        rr.Image(rgb_image2),
    )
    
    # Log points in 3D
    rr.log(
        f"{base_name}/points3d",
        rr.Points3D(
            positions=masked_pts.reshape(-1, 3),
            colors=masked_colors.reshape(-1, 3),
        ),
    )
    
    # Log predicted points
    rr.log(
        f"{base_name}/pred_points3d",
        rr.Points3D(
            positions=pred_masked_pts.reshape(-1, 3),
            colors=masked_colors.reshape(-1, 3),
        ),
    )

    rr.log(
        f"{base_name}/pointmap1",
        rr.Points3D(
            positions=pointmap1.reshape(-1, 3),
            colors=masked_colors.reshape(-1, 3),
        ),
    )
    rr.log(
        f"{base_name}/pointmap2",
        rr.Points3D(
            positions=pointmap2.reshape(-1, 3),
            colors=masked_colors.reshape(-1, 3),
        ),
    )

    # Log transformed points by GT flow
    rr.log(
        f"{base_name}/gt_transformed_points",
        rr.Points3D(
            positions=(masked_pts + masked_gt_flow).reshape(-1, 3),
            colors=masked_colors.reshape(-1, 3),
        ),
    )
    
    # Log transformed points by predicted flow
    rr.log(
        f"{base_name}/pred_transformed_points",
        rr.Points3D(
            positions=(masked_pts + masked_pred_flow).reshape(-1, 3),
            colors=masked_colors.reshape(-1, 3),
        ),
    )
    
    # Log ground truth flow
    rr.log(
        f"{base_name}/gt_sceneflow",
        rr.Arrows3D(
            origins=masked_pts.reshape(-1, 3),
            vectors=masked_gt_flow.reshape(-1, 3),
            colors=np.full((masked_pts.shape[0], 3), [0.0, 0.8, 0.2]),  # Green for ground truth
        )
    )
    
    # Log predicted flow
    rr.log(
        f"{base_name}/pred_sceneflow",
        rr.Arrows3D(
            origins=masked_pts.reshape(-1, 3),
            vectors=masked_pred_flow.reshape(-1, 3),
            colors=np.full((masked_pts.shape[0], 3), [0.8, 0.2, 0.0]),  # Red for predictions
        )
    )
    
    # Log error vectors (difference between GT and predicted)
    rr.log(
        f"{base_name}/error_vectors",
        rr.Arrows3D(
            origins=(masked_pts + masked_gt_flow).reshape(-1, 3),  # Start from GT endpoint
            vectors=(masked_pred_flow - masked_gt_flow).reshape(-1, 3),  # Vector from GT to prediction
            colors=np.full((masked_pts.shape[0], 3), [0.0, 0.2, 0.8]),  # Blue for error
        )
    )
    
    # If depth map is available, log it
    if "depthmap" in view1:
        rr.log(
            f"{base_name}/pinhole/depth",
            rr.DepthImage(view1["depthmap"]),
        )


def get_args_parser():
    """
    Parse command line arguments
    
    Returns:
        args: parsed arguments
        raft_args: parsed arguments for RAFT
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Scene Flow Evaluation with Monst3R")
    
    # RAFT model arguments
    parser.add_argument('--cfg', 
                        help='RAFT experiment configure file name', 
                        type=str, 
                        default='/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/benchmarking/SEA-RAFT/config/eval/spring-M.json')
    parser.add_argument('--model', 
                        help='RAFT checkpoint path', 
                        type=str, 
                        default='/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/benchmarking/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth')
    
    # Monst3R model arguments
    parser.add_argument('--monst3r_model', 
                        help='Monst3R model checkpoint path', 
                        type=str, 
                        default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    
    # General arguments
    parser.add_argument('--output_dir', 
                        help='Output directory', 
                        default='scene_flow_output', 
                        type=str)
    parser.add_argument('--sample_ratio', 
                        help='Ratio of points to visualize', 
                        default=1, 
                        type=float)
    
    # RAFT-specific arguments
    parser.add_argument('--scale', 
                        help='Scale factor for RAFT', 
                        default=0, 
                        type=int)
    parser.add_argument('--iters', 
                        help='Number of iterations for RAFT', 
                        default=12, 
                        type=int)
    
    # Dataset arguments
    parser.add_argument('--dataset',
                        help='Dataset type (vkitti2_multi_view_motion or stereo4d_multi_view_motion or pd4d_multi_view_motion)',
                        default="tapvid3d_multi_view_motion", 
                        type=str)
    parser.add_argument('--root_dir', 
                        help='Root directory for dataset',
                        default="/ocean/projects/cis220039p/mdt2/datasets/dydust3r", 
                        type=str)
    parser.add_argument('--num_of_views', 
                        help='Number of views',
                        default=2, 
                        type=int)
    parser.add_argument('--img_width',
                        help='Image width',
                        default=518,
                        type=int)
    parser.add_argument('--img_height',
                        help='Image height',
                        default=336,
                        type=int)
    
    # Monst3R-specific arguments
    parser.add_argument('--size', 
                        help='Shape that input images will be rescaled to for Monst3R', 
                        default=512, 
                        type=int)
    parser.add_argument('--vis_threshold', 
                        help='Visualization threshold for the point cloud viewer', 
                        default=1.5, 
                        type=float)
    
    # Metric threshold arguments
    parser.add_argument('--var_min', 
                        help='Min variance', 
                        default=-10.0, 
                        type=float)
    parser.add_argument('--var_max', 
                        help='Max variance', 
                        default=10.0, 
                        type=float)
    
    # Visualization argument
    parser.add_argument('--viz', 
                        help='Enable visualization',
                        action="store_true")
    
    # Add rerun visualization arguments
    script_add_rerun_args(parser)

    # Parse arguments
    args = parser.parse_args()
    
    # Get RAFT-specific arguments
    raft_args = sea_raft_parse_args(parser)

    return args, raft_args


def main():
    """Main function for scene flow evaluation with Monst3R"""
    # Set random seed for reproducibility
    seed_everything(0)
    
    # Parse arguments
    args, raft_args = get_args_parser()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup dataset
    if args.dataset == "vkitti2_multi_view_motion":
        dataset = VKITTI2MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=f"{args.root_dir}/VirtualKITTI2",
            resolution=(args.img_width, args.img_height),
            aug_crop=16,
            aug_monocular=False,
            transform='imgnorm',
            data_norm_type="dinov2",
            iterate_over_scenes=False,
        )
    elif args.dataset == "stereo4d_multi_view_motion":
        dataset = Stereo4DMultiView_Motion(
            num_views=args.num_of_views,
            split="test",
            ROOT=f"{args.root_dir}/stereo4d_processed",
            resolution=(args.img_width, args.img_height),
            aug_crop=16,
            aug_monocular=False,
            transform='imgnorm',
            data_norm_type="dinov2",
            iterate_over_scenes=False,
        )
    elif args.dataset == "pd4d_multi_view_motion":
        dataset = PD4DMultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=f"{args.root_dir}/paralleldomain4d",
            resolution=(args.img_width, args.img_height),
            aug_crop=16,
            aug_monocular=False,
            transform='imgnorm',
            data_norm_type="dinov2",
            iterate_over_scenes=False,
        )
    elif args.dataset == "tapvid3d_multi_view_motion":
        dataset = TAPVID3DMultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT='/ocean/projects/cis220039p/mdt2/datasets/motion_dust3r/tapvid3d/tapvid3d_dataset/adt', #f"{args.root_dir}/tapvid3d_processed",
            resolution=(args.img_width, args.img_height),
            aug_crop=16,
            aug_monocular=False,
            transform='imgnorm',
            data_norm_type="dinov2",
            iterate_over_scenes=False,
        )
    else:
        raise ValueError("Unsupported dataset type. Please choose 'vkitti2_multi_view_motion', 'stereo4d_multi_view_motion', or 'pd4d_multi_view_motion'.")    

    print(dataset.get_stats())
    
    # Setup rerun visualization if requested
    if args.viz:
        rr.script_setup(args, f"Monst3R_SEARAFT_Benchmarking_{args.dataset}")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
    
    # Set up models
    raft_model = setup_optical_flow_model(raft_args, args.model, device)
    monst3r_model = setup_monst3r_model(args.monst3r_model, device)
    
    # Initialize metrics
    metrics = {
        'epe3d_image_all': 0.0,      # Sum of per-image average EPE3D
        'epe3d_pixel_all': 0.0,      # Sum of all pixel EPE3D values
        'epe3d_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo frame
        'epe3d_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo frame
        'num_valid_pixels': 0.0,     # Count of valid pixels
        'delta_0.005': 0.0,          # Count of pixels with EPE3D < 0.005
        'delta_0.01': 0.0,           # Count of pixels with EPE3D < 0.01
        'delta_0.025': 0.0,          # Count of pixels with EPE3D < 0.025
        'delta_0.05': 0.0,           # Count of pixels with EPE3D < 0.05
        'delta_0.1': 0.0,            # Count of pixels with EPE3D < 0.1
        'delta_1': 0.0,              # Count of pixels with EPE3D < 1.0
        'outlier_0.25': 0.0          # Count of outliers with EPE3D > 0.25
    }
    
    # Process each sample in the dataset
    for idx in tqdm(range(len(dataset)), desc="Evaluating Scene Flow"):
        # Get views from dataset
        view1, view2 = dataset[idx]
        
        # Get image paths
        im1_path = view1["label"]
        im2_path = view2["label"]

        # Predict optical flow using RAFT
        flow, _ = predict_optical_flow(raft_model, raft_args, im1_path, im2_path, device)
        
        # Predict 3D points using Monst3R
        image_paths = [im1_path, im2_path]
        pointmaps, depth_conf, poses = predict_3d_points_monst3r(monst3r_model, image_paths, args.size, device)
        
        # Convert to numpy if they're not already
        pointmap1 = pointmaps[0] if isinstance(pointmaps[0], np.ndarray) else pointmaps[0].cpu().numpy()
        pointmap2 = pointmaps[1] if isinstance(pointmaps[1], np.ndarray) else pointmaps[1].cpu().numpy()
        
        # Resize pointmaps to match optical flow resolution if needed
        if pointmap1.shape[:2] != flow.shape[:2]:
            resized_pointmap1 = cv2.resize(pointmap1, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_LINEAR)
            resized_pointmap2 = cv2.resize(pointmap2, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            resized_pointmap1 = pointmap1
            resized_pointmap2 = pointmap2
        
        # Calculate scene flow
        pred_scene_flow, pred_valid_mask = calculate_scene_flow(flow, resized_pointmap1, resized_pointmap2)
        
        # Get ground truth scene flow and masks
        gt_scene_flow = view1["scene_flow"]
        gt_allo_scene_flow = view1["allo_scene_flow"]
        gt_valid_mask = view1["valid_mask"]
        gt_pts3d = view1["pts3d"]
        cam = np.linalg.inv(view1["camera_pose"])@ view2["camera_pose"]

        # Combine masks (valid only if both prediction and ground truth are valid)
        gt_valid_mask = np.logical_and(gt_valid_mask, pred_valid_mask)
        
        # Normalize point clouds for fair comparison
        norm_pred_points, pred_norm_factor = normalize_multiple_pointclouds(
            [resized_pointmap1], [gt_valid_mask]
        )
        norm_gt_points, gt_norm_factor = normalize_multiple_pointclouds(
            [gt_pts3d], [gt_valid_mask]
        )
        
        # Set thresholds for accuracy metrics
        delta_0_005_threshold = 0.005
        delta_0_01_threshold = 0.01
        delta_0_025_threshold = 0.025
        delta_0_05_threshold = 0.05
        delta_0_1_threshold = 0.1
        delta_1_threshold = 1
        outlier_0_25_threshold = 0.025
        
        # Normalize scene flow
        norm_pred_scene_flow = pred_scene_flow / pred_norm_factor
        scaled_pred_points = resized_pointmap1 / pred_norm_factor
        scaled_cam = cam.copy()
        scaled_cam[:3, 3] /= gt_norm_factor
        norm_pred_allo_scene_flow = convert_sceneflow_ego_to_allo(norm_pred_scene_flow, resized_pointmap1/pred_norm_factor, scaled_cam)

        norm_gt_scene_flow = gt_scene_flow / gt_norm_factor
        norm_gt_allo_scene_flow = gt_allo_scene_flow / gt_norm_factor
        
        # Calculate 3D end-point-error (EPE) on valid pixels
        epe3d = np.linalg.norm(norm_pred_scene_flow[gt_valid_mask] - norm_gt_scene_flow[gt_valid_mask], axis=-1)
        epe3d_norm = epe3d#/np.linalg.norm(norm_gt_scene_flow[gt_valid_mask]+1e-6, axis=-1)
        
        epe3d_allo = np.linalg.norm(norm_pred_allo_scene_flow[gt_valid_mask] - norm_gt_allo_scene_flow[gt_valid_mask], axis=-1)
        epe3d_allo_norm = epe3d_allo#/np.linalg.norm(norm_gt_allo_scene_flow[gt_valid_mask]+1e-6, axis=-1)

        # Calculate delta metrics (accuracy at different thresholds)
        delta_0_005 = np.count_nonzero(epe3d_norm < delta_0_005_threshold)
        delta_0_01 = np.count_nonzero(epe3d_norm < delta_0_01_threshold)
        delta_0_025 = np.count_nonzero(epe3d_norm < delta_0_025_threshold)
        delta_0_05 = np.count_nonzero(epe3d_norm < delta_0_05_threshold)
        delta_0_1 = np.count_nonzero(epe3d_norm < delta_0_1_threshold)
        delta_1 = np.count_nonzero(epe3d_norm < delta_1_threshold)
        
        # Calculate outlier metric (EPE > 0.25)
        outlier_0_25 = np.count_nonzero(epe3d_norm > outlier_0_25_threshold)

        # Update metrics
        metrics["epe3d_image_all"] += np.mean(epe3d)
        metrics["num_valid_pixels"] += np.sum(gt_valid_mask)
        metrics["epe3d_pixel_all"] += np.sum(epe3d)
        metrics["epe3d_allo_image_all"] += np.mean(epe3d_allo)
        metrics["epe3d_allo_pixel_all"] += np.sum(epe3d_allo)
        metrics["delta_0.05"] += delta_0_05
        metrics["delta_0.1"] += delta_0_1
        metrics["delta_1"] += delta_1
        metrics["outlier_0.25"] += outlier_0_25
        metrics["delta_0.005"] += delta_0_005
        metrics["delta_0.01"] += delta_0_01
        metrics["delta_0.025"] += delta_0_025
    
        # Print intermediate results
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"\nResults after sample {idx + 1}:")
            print(f"EPE3D (per image): {metrics['epe3d_image_all'] / (idx + 1):.4f}")
            print(f"EPE3D (per pixel): {metrics['epe3d_pixel_all'] / metrics['num_valid_pixels']:.4f}")
            print(f"EPE3D (per allo image): {metrics['epe3d_allo_image_all'] / (idx + 1):.4f}")
            print(f"EPE3D (per allo pixel): {metrics['epe3d_allo_pixel_all'] / metrics['num_valid_pixels']:.4f}")
            print(f"Delta < 0.005: {metrics['delta_0.005'] / metrics['num_valid_pixels']:.4f}")
            print(f"Delta < 0.01: {metrics['delta_0.01'] / metrics['num_valid_pixels']:.4f}")
            print(f"Delta < 0.025: {metrics['delta_0.025'] / metrics['num_valid_pixels']:.4f}")
            print(f"Delta < 0.05: {metrics['delta_0.05'] / metrics['num_valid_pixels']:.4f}")
            print(f"Delta < 0.10: {metrics['delta_0.1'] / metrics['num_valid_pixels']:.4f}")
            print(f"Delta < 1.00: {metrics['delta_1'] / metrics['num_valid_pixels']:.4f}")
            print(f"Outliers > 0.25: {metrics['outlier_0.25'] / metrics['num_valid_pixels']:.4f}")

        # Visualization
        if args.viz and (idx < 5 or idx % 50 == 0):  # Visualize first 5 and then every 50th sample
            image1 = rgb(view1["img"], norm_type="dinov2")
            image2 = rgb(view2["img"], norm_type="dinov2")
            # Set time for correct sequence in visualization
            rr.set_time_seconds("stable_time", idx)
            
            # Log images and scene flow
            base_name = f"world/view"
            rr.log(f"{base_name}/rgb_image", rr.Image(image1))
            
            # import pdb; pdb.set_trace()
            log_scene_flow_to_rerun(
                base_name,
                view1,
                view2,
                gt_pts3d,
                norm_pred_points[0],
                resized_pointmap1,
                resized_pointmap2,
                norm_gt_scene_flow,
                norm_pred_scene_flow,
                gt_valid_mask,
                image1,
                image2,
                sample_ratio=args.sample_ratio
            )


    # Calculate final metrics
    num_samples = len(dataset)
    epe3d_image_avg = metrics["epe3d_image_all"] / num_samples
    epe3d_pixel_avg = metrics["epe3d_pixel_all"] / metrics["num_valid_pixels"]
    epe3d_allo_image_avg = metrics["epe3d_allo_image_all"] / num_samples
    epe3d_allo_pixel_avg = metrics["epe3d_allo_pixel_all"] / metrics["num_valid_pixels"]
    delta_0_005_avg = metrics["delta_0.005"] / metrics["num_valid_pixels"]
    delta_0_01_avg = metrics["delta_0.01"] / metrics["num_valid_pixels"]
    delta_0_025_avg = metrics["delta_0.025"] / metrics["num_valid_pixels"]
    delta_0_05_avg = metrics["delta_0.05"] / metrics["num_valid_pixels"]
    delta_0_1_avg = metrics["delta_0.1"] / metrics["num_valid_pixels"]
    delta_1_avg = metrics["delta_1"] / metrics["num_valid_pixels"]
    outlier_0_25_avg = metrics["outlier_0.25"] / metrics["num_valid_pixels"]
    
    # Print final results
    print("\n" + "="*50)
    print("Scene Flow Evaluation Results")
    print("="*50)
    print(f"Average EPE3D per image: {epe3d_image_avg:.4f}")
    print(f"Average EPE3D per pixel: {epe3d_pixel_avg:.4f}")
    print(f"Average EPE3D per allo image: {epe3d_allo_image_avg:.4f}")
    print(f"Average EPE3D per allo pixel: {epe3d_allo_pixel_avg:.4f}")
    print(f"Acc < 0.005: {delta_0_005_avg:.4f}")
    print(f"Acc < 0.01: {delta_0_01_avg:.4f}")
    print(f"Acc < 0.025: {delta_0_025_avg:.4f}")
    print(f"Acc < 0.05: {delta_0_05_avg:.4f}")
    print(f"Acc < 0.10: {delta_0_1_avg:.4f}")
    print(f"Acc < 1.00: {delta_1_avg:.4f}")
    print(f"Number of valid pixels: {metrics['num_valid_pixels']:.0f}")
    print(f"Outliers > 0.25: {outlier_0_25_avg:.4f}")
    print("="*50)

    # Get APD metrics
    APD = np.mean([
        delta_0_005_avg,
        delta_0_01_avg,
        delta_0_025_avg,
    ])
    print(f"APD: {APD:.4f}")
    print("="*50)

    # Clean up visualization if enabled
    if args.viz:
        rr.script_teardown(args)
    
    return {
        "epe3d_image": epe3d_image_avg,
        "epe3d_pixel": epe3d_pixel_avg,
        "epe3d_allo_image": epe3d_allo_image_avg,
        "epe3d_allo_pixel": epe3d_allo_pixel_avg,
        "acc_0.005": delta_0_005_avg,
        "acc_0.01": delta_0_01_avg,
        "acc_0.025": delta_0_025_avg,
        "acc_0.05": delta_0_05_avg,
        "acc_0.1": delta_0_1_avg,
        "acc_1.0": delta_1_avg,
        "outlier_0.25": outlier_0_25_avg
    }


if __name__ == "__main__":
    main()