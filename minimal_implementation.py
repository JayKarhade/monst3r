import argparse
import os
import torch
import numpy as np
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb
import matplotlib.pyplot as pl

torch.backends.cuda.matmul.allow_tf32 = True


def get_3D_model_from_scene(outdir, scene, min_conf_thr=3, as_pointcloud=False, 
                           mask_sky=False, clean_depth=False, transparent_cams=False, 
                           cam_size=0.05, show_cam=True):
    """Extract 3D model (glb file) from a reconstructed scene"""
    if scene is None:
        return None
    
    # Post process
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # Get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    msk = to_numpy(scene.get_masks())
    
    # Camera colors
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]

    import pdb; pdb.set_trace()

    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, 
                                      as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, 
                                      cam_size=cam_size, show_cam=show_cam, silent=True,
                                      cam_color=cam_color)


def get_reconstructed_scene(image_dir, output_dir, model, device='cuda', image_size=512,
                           schedule='linear', niter=300, min_conf_thr=1.1, 
                           scenegraph_type='swinstride', winsize=5, 
                           shared_focal=True, temporal_smoothing_weight=0.01,
                           translation_weight=1.0, batch_size=16):
    """
    Main function to reconstruct scene from images in a directory
    
    Args:
        image_dir: Path to directory containing images
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
    if os.path.isdir(image_dir):
        filelist = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir))]
    else:
        filelist = [image_dir]  # Single image or video file
    
    imgs = load_images(filelist, size=image_size, verbose=True)
    
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 3D model
    outfile = get_3D_model_from_scene(output_dir, scene, min_conf_thr, 
                                     as_pointcloud=True, clean_depth=True)
    
    # Save additional outputs
    scene.save_tum_poses(f'{output_dir}/poses.txt')
    scene.save_intrinsics(f'{output_dir}/intrinsics.txt')
    scene.save_depth_maps(output_dir)
    scene.save_rgb_imgs(output_dir)
    
    return scene, outfile


def main():
    parser = argparse.ArgumentParser(description='MonST3R Scene Reconstruction')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--weights', type=str, 
                       default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth',
                       help='Path to model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for processing')
    parser.add_argument('--niter', type=int, default=300,
                       help='Number of optimization iterations')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.weights}")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(args.device)
    
    # Run reconstruction
    print(f"Processing images from {args.input_dir}")
    scene, outfile = get_reconstructed_scene(
        image_dir=args.input_dir,
        output_dir=args.output_dir,
        model=model,
        device=args.device,
        image_size=args.image_size,
        niter=args.niter,
        batch_size=args.batch_size
    )
    
    print(f"Reconstruction completed!")
    print(f"3D model saved to: {outfile}")
    print(f"Additional outputs saved in: {args.output_dir}")


if __name__ == '__main__':
    main()