import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
from tqdm import tqdm
import rerun as rr
import glob
from natsort import natsorted
import json
import copy

# Import CoTrackerv3

from cotracker.predictor import CoTrackerPredictor

# Import Monst3R
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Import dataset modules
from anymap.datasets.multi_view_motion.tapvid3d_pstudio import TAPVID3D_PStudio_MultiView_Motion
from anymap.datasets.multi_view_motion.tapvid3d_drivetrack import TAPVID3D_DriveTrack_MultiView_Motion
from anymap.datasets.multi_view_motion.tapvid3d_adt import TAPVID3D_ADT_MultiView_Motion
from anymap.datasets.multi_view_motion.kubric_eval import KubricEval
from anymap.datasets.multi_view_motion.dynamic_replica_eval import DynamicReplicaEval

# Import visualization utilities
from anymap.utils.viz import script_add_rerun_args
from anymap.utils.image import rgb
from anymap.utils.misc import seed_everything

def init_dataset(args, sequence_name):

    sequence_path = os.path.join(args.dataset_dir, sequence_name)

    if args.dataset == "tapvid3d_pstudio":
        dataset = TAPVID3D_PStudio_MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type="identity",
            iterate_over_scenes=False,
        )
    elif args.dataset == "tapvid3d_drivetrack":
        dataset = TAPVID3D_DriveTrack_MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type="identity",
            iterate_over_scenes=False,
        )
    elif args.dataset == "tapvid3d_adt":
        dataset = TAPVID3D_ADT_MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type="identity",
            iterate_over_scenes=False,
        )
    elif args.dataset == "kubric_eval":
        dataset = KubricEval(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type="identity",
            iterate_over_scenes=False,
        )
    elif args.dataset == "dynamic_replica_eval":
        dataset = DynamicReplicaEval(
            num_views=args.num_of_views,
            split="val",
            ROOT=args.dataset_dir,
            seq_name=sequence_name,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type="identity",
            iterate_over_scenes=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset


def log_data_to_rerun(base_name, image0, image1, poses, intrinsics, pts3d0, pts3d1, allo_scene_flow, mask0, mask1):
    """
    Log data to rerun for visualization

    Args:
        image0: first image (numpy array)
        image1: second image (numpy array)
        poses: ground truth camera poses (list of numpy arrays)
        intrinsics: camera intrinsics (numpy array)
        pts3d0: ground truth 3D points for first image (numpy array)
        pts3d1: ground truth 3D points for second image (numpy array)
        allo_scene_flow: ground truth scene flow in allo-centric frame (numpy array)
        mask0: valid mask for points (numpy array)
    """

    # Log camera info and loaded data
    height, width = image0.shape[:2]
    rr.log(
        f"{base_name}/view0",
        rr.Transform3D(
            translation=poses[0][:3, 3],
            mat3x3=poses[0][:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/view1",
        rr.Transform3D(
            translation=poses[1][:3, 3],
            mat3x3=poses[1][:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/view0/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/view1/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/view0/pinhole/rgb0",
        rr.Image(image0),
    )
    rr.log(
        f"{base_name}/view1/pinhole/rgb1",
        rr.Image(image1),
    )

    # Log points and scene flow
    filtered_pts3d0 = pts3d0[mask0]
    filtered_pts3d_col = image0[mask0]
    rr.log(
        f"{base_name}/pts3d0",
        rr.Points3D(
            positions=filtered_pts3d0.reshape(-1, 3),
            colors=filtered_pts3d_col.reshape(-1, 3),
        ),
    )

    filtered_pts3d1 = pts3d1[mask1]
    filtered_pts3d_col1 = image1[mask1]
    rr.log(
        f"{base_name}/pts3d1",
        rr.Points3D(
            positions=filtered_pts3d1.reshape(-1, 3),
            colors=filtered_pts3d_col1.reshape(-1, 3),
        ),
    )

    # Log allo-centric scene flow
    filtered_scene_flow = allo_scene_flow[mask0]
    rr.log(
        f"{base_name}/scene_flow",
        rr.Arrows3D(
            origins=filtered_pts3d0.reshape(-1, 3),
            vectors=filtered_scene_flow.reshape(-1, 3),
            # colors=filtered_pts3d_col.reshape(-1, 3),
        ),
    )


def normalize_multiple_pointclouds(pts_list, valid_masks):
    """
    Normalize multiple pointclouds using average distance to origin.
    
    Args:
        pts_list: List of point clouds, each with shape HxWx3
        valid_masks: List of masks indicating valid points
    
    Returns:
        List of normalized point clouds, normalization factor
    """
    # Collect only valid points for normalization calculation
    all_valid_pts = []
    
    for i, pts in enumerate(pts_list):
        mask = valid_masks[i]
        valid_pts = pts[mask]  # Only extract valid points
        if len(valid_pts) > 0:
            all_valid_pts.append(valid_pts)
    
    if not all_valid_pts:
        # Handle edge case where no valid points exist
        return pts_list, 1.0
    
    # Concatenate all valid points
    all_pts = np.concatenate(all_valid_pts, axis=0)
    
    # Compute average distance to origin
    all_dis = np.linalg.norm(all_pts, axis=-1)
    norm_factor = np.mean(all_dis)
    norm_factor = max(norm_factor, 1e-8)  # Prevent division by zero
    
    # Normalize each point cloud
    res = [pts / norm_factor for pts in pts_list]
    
    return res, norm_factor

def get_monst3r_reconstruction(image_paths, model, device='cuda', image_size=512,
                           schedule='linear', niter=300, min_conf_thr=1.1, 
                           scenegraph_type='swinstride', winsize=5, 
                           shared_focal=True, temporal_smoothing_weight=0.01,
                           translation_weight=1.0, batch_size=16):
    """
    Main function to reconstruct scene from images in a directory
    
    Args:
        image_paths: List of paths to images
        output_dir: Directory to save outputs
        model: Loaded MonST3R model
        device: Device to run on ('cuda' or 'cpu')
        image_size: Size to resize images to
        schedule: Learning rate schedule ('linear' or 'cosine')
        niter: Number of optimization iterations
        min_conf_thr: Minimum confidence threshold
        scenegraph_type: Type of scene graph to use
        winsize: Window size for scene graph
        shared_focal: Whether to use shared focal length
        temporal_smoothing_weight: Weight for temporal smoothing
        translation_weight: Weight for translation
        batch_size: Batch size for inference
    
    Returns:
        scene: Reconstructed scene object
        outfile: Path to output 3D model file
    """
    model.eval()
    
    # Load images    
    imgs = load_images(image_paths, size=image_size, verbose=True)

    # Handle single image case
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    
    # Set up scene graph
    if scenegraph_type in ["swin", "swinstride", "swin2stride"]:
        scenegraph_type = f"{scenegraph_type}-{winsize}-noncyclic"
    
    # Create pairs and run inference
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=True)

    # Global alignment
    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=device, mode=mode, verbose=True, 
                              shared_focal=shared_focal, 
                              temporal_smoothing_weight=temporal_smoothing_weight,
                              translation_weight=translation_weight,
                              num_total_iter=niter)
        
        # Compute global alignment
        lr = 0.01
        scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=True)
        
    # Get parameters
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()
    cams2world = scene.get_im_poses().detach().cpu().numpy()
    pts3d = scene.get_pts3d(raw_pts=True)
    pts3d = torch.stack(pts3d).detach().cpu().numpy()

    return intrinsics, cams2world, pts3d



def monst3r_cotracker(args, collected_image_paths, collected_images, collected_query_pts, device="cuda"):

    # Convert to torch
    query_pts = torch.tensor(collected_query_pts, dtype=torch.float32).to(device)[None]
    images = torch.stack(collected_images).to(device) # (N, 3, H, W)
    images = images[None]

    # First run through co-tracker
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    pred_tracks, pred_visibility = cotracker(images*255, query_pts)

    # Run images on Monst3r
    monst3r_model = AsymmetricCroCo3DStereo.from_pretrained('Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt').to(device)
    monst3r_model.eval()

    pred_intrinsics, pred_cams2world, pointmaps = get_monst3r_reconstruction(
        collected_image_paths,
        model=monst3r_model,
        device=device,
        image_size=args.image_size,
        niter=args.niter,
        batch_size=args.batch_size
    )

    # Use pred_tracks from cotracker to get 3D tracks
    pred_tracks = pred_tracks.squeeze(0)  # (T, N, 2)
    T, N, _ = pred_tracks.shape
    _, H, W, _ = pointmaps.shape
    
    # Clamp coordinates to valid range and convert to integer indices
    x_coords = torch.clamp(pred_tracks[..., 0], 0, W-1).long()  # (T, N)
    y_coords = torch.clamp(pred_tracks[..., 1], 0, H-1).long()  # (T, N)

    # Create batch indices for advanced indexing
    t_indices = torch.arange(T, device=device).unsqueeze(1).expand(-1, N)  # (T, N)

    t_indices = t_indices.cpu().numpy()
    y_coords = y_coords.cpu().numpy()
    x_coords = x_coords.cpu().numpy()

    tracks_3d = pointmaps[t_indices, y_coords, x_coords]

    # Form TxHxWx3 tracks from tracks_3d
    tracks_3d_pointmap = np.zeros((T, H, W, 3), dtype=np.float32)
    tracks_3d_pointmap[t_indices, collected_query_pts[:, 2].astype(int), collected_query_pts[:, 1].astype(int)] = tracks_3d

    result_dict = {}
    result_dict["pointmaps"] = pointmaps
    result_dict["tracks_3d_pointmap"] = tracks_3d_pointmap
    result_dict["extrinsics"] = pred_cams2world
    result_dict["intrinsics"] = pred_intrinsics

    return result_dict

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default="psc")
    parser.add_argument("--config_dir", default="/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/configs")
    parser.add_argument('--dataset', help='Dataset type', default="tapvid3d_pstudio", type=str)
    parser.add_argument("--dataset_dir", default="/ocean/projects/cis220039p/mdt2/datasets/dydust3r/tapvid3d_dataset/pstudio")
    # parser.add_argument('--dataset', help='Dataset type', default="dynamic_replica_eval", type=str)
    # parser.add_argument("--dataset_dir", default="/ocean/projects/cis220039p/mdt2/datasets/dydust3r/dynamic_replica_data", type=str)
    parser.add_argument("--img_width", default=512, type=int, help="Image width")
    parser.add_argument("--img_height", default=288, type=int, help="Image height")
    parser.add_argument("--num_of_views", default=2, type=int, help="Number of views to use")
    parser.add_argument("--viz", action="store_true", help="Visualize results using rerun")
    parser.add_argument("--max_samples", default=None, type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--sample_ratio", default=1, type=float, help="Ratio of points to visualize")
    parser.add_argument("--only_dynamic", default=True, help="Only evaluate dynamic points", action="store_true")
    
    # Monst3R arguments
    parser.add_argument('--weights', 
                        help='Monst3R model checkpoint path', 
                        type=str, 
                        default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--image_size', type=int, default=518,
                       help='Image size for processing')
    parser.add_argument('--niter', type=int, default=300,
                       help='Number of optimization iterations')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')

    # Add rerun visualization arguments
    script_add_rerun_args(parser)
    
    args = parser.parse_args()

    return args


def main():
    """Main function for scene flow evaluation"""

    # Set random seed for reproducibility
    seed_everything(0)

    # Parse arguments
    args = get_args_parser()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup rerun visualization if requested
    if args.viz:
        rr.script_setup(args, f"Monst3r_CoTracker_Benchmarking_{args.dataset}")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Initialize metrics
    all_seq_metrics = {
        'epe3d_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo-centric
        'epe3d_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo-centric
        'num_valid_pixels': 0.0,     # Count of valid pixels
        'delta_0.1': 0.0,            # Count of pixels with EPE3D < 0.1
        'delta_0.3': 0.0,            # Count of pixels with EPE3D < 0.3
        'delta_0.5': 0.0,            # Count of pixels with EPE3D < 0.5
        'outlier_0.25': 0.0          # Count of outliers with EPE3D > 0.25
    }

    # Set thresholds for accuracy metrics
    delta_thresholds = [0.1, 0.3, 0.5]
    outlier_thresholds = [0.25]

    # Dataset initialization
    if args.dataset == "tapvid3d_pstudio":
        data_sequences = natsorted(glob.glob(f"{args.dataset_dir}/*.npz"))[:5]
    elif args.dataset == "tapvid3d_adt":
        data_sequences = natsorted(glob.glob(f"{args.dataset_dir}/*multiuser*.npz"))[:5]
    elif args.dataset == "tapvid3d_drivetrack":
        data_sequences = json.load(open('/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/anymap/datasets/multi_view_motion/drive_track_eligible_scenes.json'))[:5]
    elif args.dataset == "kubric_eval":
        data_sequences = [str(i) for i in range(5700, 5700+50)][:5]
    elif args.dataset == "dynamic_replica_eval":
        data_sequences = json.load(open('/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/anymap/datasets/multi_view_motion/dynamic_replica_eval_val_scenes.json'))[:5]

    print(f"Found {len(data_sequences)} sequences.")

    for sequence_path in data_sequences:
        dataset = init_dataset(args, os.path.basename(sequence_path))

        # Temporarily go down to 50 only
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(50, len(dataset))))

        print(f"Processing sequence: {os.path.basename(sequence_path)} with {len(dataset)} pairs")


        collected_imgs = []
        collected_img_paths = []

        query_point_indices = np.where(dataset[0][0]["valid_mask"] > 0)
        query_pts = np.column_stack((np.zeros_like(query_point_indices[0]), query_point_indices[1], query_point_indices[0]))  # (0, x, y) format

        # Collect all views before hand for passing all images to model
        for idx in tqdm(range(len(dataset))):
            _, cur_view = dataset[idx]
            collected_imgs.append(cur_view["img"])
            collected_img_paths.append(cur_view["label"])


        # Run model
        predictions = monst3r_cotracker(args, collected_img_paths, collected_imgs, query_pts, device=device)

        # Initialize metrics
        cur_seq_metrics = {
            'epe3d_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo-centric
            'epe3d_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo-centric
            'num_valid_pixels': 0.0,     # Count of valid pixels
            'delta_0.1': 0.0,            # Count of pixels with EPE3D < 0.1
            'delta_0.3': 0.0,            # Count of pixels with EPE3D < 0.3
            'delta_0.5': 0.0,            # Count of pixels with EPE3D < 0.5
            'outlier_0.25': 0.0          # Count of outliers with EPE3D > 0.25
        }

        # Iterate over predictions to calculate metrics
        for idx in tqdm(range(len(dataset))):

            # Get ground truth
            view0, view1 = dataset[idx]

            im0_path = view0["label"]
            im1_path = view1["label"]
            gt_allo_scene_flow = view1["allo_scene_flow"]
            gt_valid_mask0 = view0["valid_mask"]
            gt_valid_mask1 = view1["valid_mask"]
            gt_pts3d0 = view0["pts3d"]
            gt_pts3d1 = view1["pts3d"]
            gt_cam0 = view0["camera_pose"]
            gt_cam1 = view1["camera_pose"]

            pred_pts3d = [predictions["pointmaps"][0].copy(), predictions["pointmaps"][idx].copy()]

            pred_tracks_3d_pts3d = [predictions["tracks_3d_pointmap"][0].copy(), predictions["tracks_3d_pointmap"][idx].copy()]
            pred_allo_scene_flow = pred_tracks_3d_pts3d[1] - pred_tracks_3d_pts3d[0]

            poses_c2w = [predictions["extrinsics"][0].copy(), predictions["extrinsics"][idx].copy()]
            pred_intrinsics = [predictions["intrinsics"][0].copy(), predictions["intrinsics"][idx].copy()]

            # Combine other masks to gt_valid_mask
            if args.only_dynamic:
                gt_valid_mask0 = gt_valid_mask0 & (gt_allo_scene_flow[:, :, 2] != 0)
            
            # Normalize clouds, poses and scene flow
            norm_gt_points, gt_norm_factor = normalize_multiple_pointclouds([gt_pts3d0, gt_pts3d1], [gt_valid_mask0, gt_valid_mask1])
            # norm_gt_points = [gt_pts3d0 / gt_norm_factor, gt_pts3d1 / gt_norm_factor]
            norm_gt_allo_scene_flow = gt_allo_scene_flow / gt_norm_factor
            gt_cam0[:3, 3] /= gt_norm_factor
            gt_cam1[:3, 3] /= gt_norm_factor

            norm_pred_points, pred_norm_factor = normalize_multiple_pointclouds([pred_pts3d[0], pred_pts3d[1]], [gt_valid_mask0, gt_valid_mask1])
            # norm_pred_points = [pred_pts3d[0] / pred_norm_factor, pred_pts3d[1] / pred_norm_factor]
            norm_pred_allo_scene_flow = pred_allo_scene_flow / pred_norm_factor
            poses_c2w[0][:3, 3] /= pred_norm_factor
            poses_c2w[1][:3, 3] /= pred_norm_factor


            # Calculate EPE3D
            # import pdb; pdb.set_trace()  # Debugging breakpoint
            if gt_valid_mask0.sum() == 0:
                print(f"Skipping image pair {idx} due to no valid points in gt_valid_mask0")
                continue
            # epe3d = np.linalg.norm(norm_gt_allo_scene_flow[gt_valid_mask0] - norm_pred_allo_scene_flow[gt_valid_mask0], axis=-1)
            epe3d = np.linalg.norm((norm_gt_points[0] + norm_gt_allo_scene_flow)[gt_valid_mask0] - (norm_pred_points[0] + norm_pred_allo_scene_flow)[gt_valid_mask0], axis=-1)

            cur_seq_metrics['epe3d_allo_image_all'] += np.mean(epe3d)
            cur_seq_metrics['epe3d_allo_pixel_all'] += np.sum(epe3d)
            cur_seq_metrics['num_valid_pixels'] += np.sum(gt_valid_mask0)

            # Calculate delta metrics
            for threshold in delta_thresholds:
                norm_threshold = threshold / gt_norm_factor
                cur_seq_metrics[f'delta_{threshold}'] += np.sum(epe3d < norm_threshold)

            # Calculate outlier metrics
            for threshold in outlier_thresholds:
                norm_threshold = threshold / gt_norm_factor
                cur_seq_metrics[f'outlier_{threshold}'] += np.sum(epe3d > norm_threshold)

            # Print ongoing avg epe3d_allo_image_all
            # print(cur_seq_metrics['epe3d_allo_image_all'] / (idx + 1))

            # Visualization
            if args.viz:
                viz_img0 = rgb(view0["img"], norm_type="dinov2")
                viz_img1 = rgb(view1["img"], norm_type="dinov2")

                # Set time for correct sequence in visualization
                rr.set_time_seconds("stable_time", idx)

                log_data_to_rerun(
                    base_name="world/gt",
                    image0=viz_img0,
                    image1=viz_img1,
                    poses=[gt_cam0, gt_cam1],
                    intrinsics=view0["camera_intrinsics"],
                    pts3d0=norm_gt_points[0],
                    pts3d1=norm_gt_points[1],
                    allo_scene_flow=norm_gt_allo_scene_flow,
                    mask0=gt_valid_mask0,
                    mask1=gt_valid_mask1,
                )

                # log_data_to_rerun(
                #     base_name="world/pred",
                #     image0=viz_img0,
                #     image1=viz_img1,
                #     poses=[gt_cam0, gt_cam1],
                #     intrinsics=view0["camera_intrinsics"],
                #     pts3d0=norm_gt_points[0],
                #     pts3d1=norm_gt_points[1],
                #     allo_scene_flow=norm_pred_allo_scene_flow,
                #     mask0=gt_valid_mask0,
                #     mask1=gt_valid_mask1,  # Using the same mask for both images
                # )

                log_data_to_rerun(
                    base_name="world/pred",
                    image0=viz_img0,
                    image1=viz_img1,
                    poses=[poses_c2w[0], poses_c2w[1]],
                    intrinsics=pred_intrinsics[0],
                    pts3d0=norm_pred_points[0],
                    pts3d1=norm_pred_points[1],
                    allo_scene_flow=norm_pred_allo_scene_flow,
                    mask0=None,
                    mask1=None,  # Using the same mask for both images
                )

        if args.viz:
            import pdb; pdb.set_trace()

        # Aggregate and Print sequence metrics
        cur_seq_metrics['epe3d_allo_image_all'] /= len(dataset)
        cur_seq_metrics['epe3d_allo_pixel_all'] /= cur_seq_metrics['num_valid_pixels']
        for threshold in delta_thresholds:
            cur_seq_metrics[f'delta_{threshold}'] /= cur_seq_metrics['num_valid_pixels']
        for threshold in outlier_thresholds:
            cur_seq_metrics[f'outlier_{threshold}'] /= cur_seq_metrics['num_valid_pixels']

        print(f"sequence:{sequence_path}")
        print(f"metrics: {cur_seq_metrics}")

        # Add sequence metrics into all_seq_metrics
        all_seq_metrics['epe3d_allo_image_all'] += cur_seq_metrics['epe3d_allo_image_all']
        all_seq_metrics['epe3d_allo_pixel_all'] += cur_seq_metrics['epe3d_allo_pixel_all']
        all_seq_metrics['num_valid_pixels'] += cur_seq_metrics['num_valid_pixels']
        for threshold in delta_thresholds:
            all_seq_metrics[f'delta_{threshold}'] += cur_seq_metrics[f'delta_{threshold}']
        for threshold in outlier_thresholds:
            all_seq_metrics[f'outlier_{threshold}'] += cur_seq_metrics[f'outlier_{threshold}']

    # Finalize all_seq_metrics
    all_seq_metrics['epe3d_allo_image_all'] /= len(data_sequences)
    all_seq_metrics['epe3d_allo_pixel_all'] /= len(data_sequences)
    for threshold in delta_thresholds:
        all_seq_metrics[f'delta_{threshold}'] /= len(data_sequences)
    for threshold in outlier_thresholds:
        all_seq_metrics[f'outlier_{threshold}'] /= len(data_sequences)

    print("Final Metrics across all sequences:")
    print(f"epe3d_allo_image_all: {all_seq_metrics['epe3d_allo_image_all']}")
    print(f"epe3d_allo_pixel_all: {all_seq_metrics['epe3d_allo_pixel_all']}")
    print(f"num_valid_pixels: {all_seq_metrics['num_valid_pixels']}")
    for threshold in delta_thresholds:
        print(f"delta_{threshold}: {all_seq_metrics[f'delta_{threshold}']}")
    for threshold in outlier_thresholds:    
        print(f"outlier_{threshold}: {all_seq_metrics[f'outlier_{threshold}']}")

    # Save metrics to txt
    metrics_file = os.path.join(f"{args.dataset}_vggt_cotracker_benchmarking_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Final Metrics across all sequences:\n")
        f.write(f"epe3d_allo_image_all: {all_seq_metrics['epe3d_allo_image_all']}\n")
        f.write(f"epe3d_allo_pixel_all: {all_seq_metrics['epe3d_allo_pixel_all']}\n")
        f.write(f"num_valid_pixels: {all_seq_metrics['num_valid_pixels']}\n")
        for threshold in delta_thresholds:
            f.write(f"delta_{threshold}: {all_seq_metrics[f'delta_{threshold}']}\n")
        for threshold in outlier_thresholds:
            f.write(f"outlier_{threshold}: {all_seq_metrics[f'outlier_{threshold}']}\n")

    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()