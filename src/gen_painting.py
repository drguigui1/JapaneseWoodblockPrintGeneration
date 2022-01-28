# -*- coding: utf-8 -*-

# parsing arguments
from argparse import ArgumentParser

# OpenCV (save/load and Canny edge detector)
import cv2

# Pytorch
import torch

import numpy as np

# Torchvision transform
from torchvision import transforms

# Generator model
from pix2pix import Generator

# =====================
# ===== READ/SAVE =====
# =====================

def read_img(img_path):
    '''
    Read the image

    Parameters:
    ----------
    img_path: Path to the image to read

    Returns:
    ----------
    Numpy array corresponding to the image
    '''
    return cv2.imread(img_path)

def save_img(img_path, img):
    '''
    Save the result image

    Parameters:
    ----------
    img_path: Path to save the image
    img: Image to save (numpy array)
    '''
    cv2.imwrite(img_path, img)

# ====================
# ======= MODEL ======
# ====================

# In this script we only use the Generator model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path, model):
    '''
    Load model using the checkpoint

    Parameters:
    ----------
    path: Path to the checkpoint
    model: Pytorch model (load the checkpoint parameters in the model)

    Returns:
    ----------
    Pytorch model with proper parameters
    '''
    print("=> Loading checkpoint")
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    # Set the model in avaluation mode
    model.eval()

    return model

def process_img(img):
    '''
    Process the image before the model prediction
    - Convert to gray scale
    - Apply Canny edge detector
    - Apply threshold to keep only relevant information
    - Normalize data
    - Stack image for 3 channel

    Parameters:
    ----------
    img: Input image to process

    Returns:
    ----------
    The processed image
    '''
    # Convert input data to grayscale and uint8
    img_gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    img_gray = img_gray.astype(np.uint8)

    # Get edges using Canny edges detector
    edges = cv2.Canny(img_gray, 150, 220)

    save_img('edges.jpg', edges)

    # Get input and target (in torch tensor)
    # Apply a threshold to keep only relevant/main information
    input = torch.from_numpy((edges > 125).astype(float))

    # Get the transformation object (for normalization)
    transform = transforms.Compose([
                        transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize(mean=0.5,
                                            std=0.5)
                    ])

    # Stack the input to get a 3 channel image (Still grayscale)
    input = torch.stack([input, input, input], dim=0)
    input = input.reshape((1, input.shape[0], input.shape[1], input.shape[2]))

    # Apply transformation on input and on target image
    return transform(input)

def prediction(gen, img):
    '''
    Make the prediction, call the generator

    Parameters:
    ----------
    gen: Generator model
    img: Input image (drawing / picture image)

    Returns:
    ----------
    Torch tensor, with the prediction
    '''
    with torch.no_grad():
        img_input = process_img(img)
        img_input = img_input.to(DEVICE)

        pred = gen(img_input)

        # Remove normalization
        pred = pred * 0.5 + 0.5

        pred = torch.permute(pred, (0, 2, 3, 1))

    # Put pixels in proper range
    pred = pred.detach().cpu().numpy()
    pred *= 255.0
    pred = pred.astype(np.uint8)
    pred = pred.squeeze()

    return cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

# ====================
# ======= MAIN =======
# ====================

def build_arg_list():
    '''
    Build the list of arguments

    Returns:
    ----------
    Argument parser object
    '''
    arg_parser = ArgumentParser(description='Launch model generation')
    arg_parser.add_argument('path', help='Path to the original image')
    return arg_parser.parse_args()

def main():
    arg_list = build_arg_list()

    # Load image
    print("1. Loading image")
    img = read_img(arg_list.path)

    # Create the model
    print("2. Create the model")
    gen = Generator()

    # Load pretrained weights
    print("3. Load checkpoint")
    checkpoint = 'model/gen.pth.tar'
    load_model(checkpoint, gen)

    # make the prediction
    print("4. Make prediction")
    pred = prediction(gen, img).squeeze()

    # save img
    print("5. Save result")
    save_img('save.png', pred)

if __name__ == "__main__":
    main()
