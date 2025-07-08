from demo_utils.utils import (load_modules, 
                              load_sample_from_dataset,
                              load_sample_from_path, 
                              inference_3d, 
                              inference_rendered_2d)

import argparse
import os
from torchvision.utils import save_image

def main():
    parser = argparse.ArgumentParser(description='SceneDINO demo script')
    parser.add_argument('--ckpt', type=str, default='out/scenedino-pretrained/seg-best-dino/', help='path to the checkpoint (default: out/scenedino-pretrained/seg-best-dino/)')
    parser.add_argument('--image', type=str, help='demo image path (default: first image of test set)')
    args = parser.parse_args()
    
    # Load model, ray sampler, datasets
    ckpt_name = "checkpoint.pt"
    net, renderer, ray_sampler, test_dataset = load_modules(args.ckpt, ckpt_name)

    if args.image:
        # Load image from path
        images, poses, projs = load_sample_from_path(args.image, intrinsic=None)
    else:
        # Load image from test dataset
        images, poses, projs = load_sample_from_dataset(0, test_dataset)

    # Encode input image
    net.encode(images, projs, poses, ids_encoder=[0])
    net.set_scale(0)

    # Inference rendered back into 2D (input view, can also be any other though)
    dino_full_2d, depth_2d, seg_2d = inference_rendered_2d(net, poses, projs, ray_sampler, renderer)
    print("\n--- 2D Inference ---")
    print("Rendered Features: ", dino_full_2d.size())
    print("Depth:             ", depth_2d.size())
    if seg_2d is not None:
        print("Segmentation:      ", seg_2d.size())
    else:
        print("-> No SSC head linked.")

    # Fit PCA for visualization
    net.encoder.fit_visualization(dino_full_2d.flatten(0, -2))

    # Save images + rendered features
    if not os.path.exists("out/demo-out"):
        os.mkdir("out/demo-out")
    save_image(images.squeeze()/2 + 0.5, "out/demo-out/input_image.png")

    dino_pca_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=0).permute(2,0,1)
    dino_pca_2d -= dino_pca_2d.min()
    save_image(dino_pca_2d / dino_pca_2d.max(), "out/demo-out/feat_pca_00_02.png")

    dino_pca_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=3).permute(2,0,1)
    dino_pca_2d -= dino_pca_2d.min()
    save_image(dino_pca_2d / dino_pca_2d.max(), "out/demo-out/feat_pca_03_05.png")

    dino_pca_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=6).permute(2,0,1)
    dino_pca_2d -= dino_pca_2d.min()
    save_image(dino_pca_2d / dino_pca_2d.max(), "out/demo-out/feat_pca_06_08.png")

    # Inference in 3D grid
    x_range = (-10, 10)
    y_range = (-5, 5)
    z_range = (0, 20)
    resolution = 0.2
    xyz, dino_full_3d, sigma_3d, seg_3d = inference_3d(net, x_range, y_range, z_range, resolution)
    print("\n--- 3D Inference ---")
    print("Features:     ", dino_full_3d.size())
    print("Vol. Density: ", sigma_3d.size())
    if seg_3d is not None:
        print("SSC:          ", seg_3d.size())
    else:
        print("-> No SSC head linked.")

    # Not optimal, fitting PCA on surfaces would be better
    # net.encoder.fit_visualization(dino_full_3d.flatten(0, -2))  
    dino_pca_3d = net.encoder.transform_visualization(dino_full_3d, from_dim=0)


if __name__ == "__main__":
    main()
