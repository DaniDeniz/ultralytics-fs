import cv2
import numpy as np
import torch


def convert_image_to_ycbcr(img):
    img = np.transpose(img[::-1], (1, 2, 0)) # First, RGB to BGR ([::-1]), then change shape format (H,W,3) to (3,H,W)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(img)
    img = cv2.merge((y, cb, cr))
    img = np.transpose(img, (2, 0, 1))  # change shape format (3, H, W) to (H, W, 3)
    return img


def convert_to_ycbcr(img):
    is_tensor = torch.is_tensor(img)
    if is_tensor:
        img = img.numpy()

    if len(img.shape) > 3:
        for i in range(img.shape[0]):
            img[i] = convert_image_to_ycbcr(img[i])
    else:
        img = convert_image_to_ycbcr(img)

    if is_tensor:
        img = torch.from_numpy(np.ascontiguousarray(img))
    return img

