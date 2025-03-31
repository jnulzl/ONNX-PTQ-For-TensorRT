import os
import sys
import cv2
import numpy as np
import argparse

from utils import pre_process


def main():
    """Prepares calibration data from ImageNet dataset and saves input dictionary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration_data_list_file_path", type=str, default="", help="Path to calibration data path."
    )    
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=512,
        help="Number[1-100000] of images to use in calibration.",
    )
    parser.add_argument(
        "--shape", nargs="+", 
        type=int, 
        default=[640, 640], 
        help="inference size h,w"
    )    
    parser.add_argument(
        "--des_channel",
        type=int,
        default=3,
        help="The channels of des image.",
    )    
    parser.add_argument(
        "--means", nargs="+", 
        type=float, 
        default=[0.0], 
        help="mean value"
    )
    parser.add_argument(
        "--scales", nargs="+", 
        type=float, 
        default=[1.0], 
        help="scale value"
    )
    
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to save the image tensor data in FP16 format."
    )
    parser.add_argument(
        "--is_fixed_scale", action="store_true", help="Whether to letterbox resize."
    )
    parser.add_argument(
        "--is_bgr2rgb", action="store_true", help="Whether to bgr2rgb."
    )
    parser.add_argument(
        "--is_hwc2chw", action="store_true", help="Whether to hwc2chw."
    )    
    parser.add_argument(
        "--output_path", type=str, default="calib.npy", help="Path to output npy file."
    )
        
    args = parser.parse_args()
    
    print(args)
    if not os.path.exists(args.calibration_data_list_file_path):
        raise Exception("%s not exists!"%(args.calibration_data_list_file_path))
        
    with open(args.calibration_data_list_file_path, 'r') as fpR:
        img_list = fpR.readlines()

    calibration_data_size = min(args.calibration_data_size , len(img_list))
    img_list = img_list[:calibration_data_size]
    calib_tensor = []
    for img_path in img_list:
        img_path = img_path.strip()
        print("%s"%(img_path))
        if not os.path.exists(img_path):
            print("Passing %s"%(img_path))
            continue
        img = cv2.imread(img_path)
        image = pre_process(img, 
                            des_channel=args.des_channel,
                            new_shape=args.shape,
                            is_fixed_scale=args.is_fixed_scale,
                            means=args.means,
                            scales=args.scales,
                            is_bgr2rgb=args.is_bgr2rgb,
                            is_hwc2chw=args.is_hwc2chw,
                            is_to_fp16=args.fp16)
    
        calib_tensor.append(image)
    if 0 != len(calib_tensor):
        calib_tensor = np.stack(calib_tensor, axis=0)
        print("calib_tensor.shape : ", calib_tensor.shape)
        np.save(args.output_path, calib_tensor)


if __name__ == '__main__': 
    main()
