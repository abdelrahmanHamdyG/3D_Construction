from generate_mask import run_mask_generation
from processing import resize_images
from generating_depth_maps import run_depth_generation
from super_glue import run_keypoint_matching
from full_construction import run_construction
from surReconstruction import do_surface_construction



def main():
    
    print("\nStarting image processing...")
    resize_images()
    print("Image resizing completed.\n\n")

    print("\n\nStarting mask generation...")
    
    run_mask_generation()
    print("Mask generation completed.\n\n")

    print("\n\nStarting depth map generation...")
    run_depth_generation()
    print("Depth map generation completed.\n\n")

    print("\n\nStarting keypoint matching...")
    run_keypoint_matching()
    print("Keypoint matching completed.\n\n")
    
    print("\n\nStarting 3D construction...")
    run_construction()
    print("3D construction completed.\n\n")

    print("\n\nStarting surface reconstruction...")
    do_surface_construction()
    print("Surface reconstruction completed.\n\n")

    

main()
