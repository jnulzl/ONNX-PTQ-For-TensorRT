import os
import sys
import cv2
import numpy as np
import argparse

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # return im, ratio, (dw, dh)
    return im


def pre_process(
    img, # HWC
    des_channel=3,
    new_shape=[640, 640],
    is_fixed_scale=True, 
    means=[0., 0., 0.], 
    scales=[1., 1., 1.],
    is_bgr2rgb=True,
    is_hwc2chw=True,
    is_to_fp16=False
    ):
    
    shape = img.shape
    if 2 == len(shape) and 3 == des_channel:
        img = cv2.cvtColor(cv2.COLOR_GRAY2RGB)
    elif 3 == len(shape) and 1 == shape[-1] and 3 == des_channel:
        img = cv2.cvtColor(cv2.COLOR_GRAY2RGB)
    else:
        pass
    
    if is_fixed_scale:
        img = letterbox(img, new_shape)
    else:
        img = cv2.resize(img, new_shape)
    
    if is_bgr2rgb:
        img = img[:,:,::-1]
        
    img = img - np.array(means)
    img *= np.array(scales)
    
    if is_hwc2chw:
        img = img.transpose(2, 0, 1)
    
    return img.astype(np.float16) if is_to_fp16 else img.astype(np.float32)

