# -*- coding: utf-8 -*-

import cv2
import glob
import os
import sys
from pathlib import Path

def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_img(img_path):
    return cv2.imread(img_path)

def save_img(img_path, img):
    cv2.imwrite(img_path, img)

def resize_img(img):
    nb_rows, nb_cols, _ = img.shape
    if nb_rows < nb_cols:
        # Get factor to go from nb_rows to nb_cols
        factor = nb_cols / nb_rows

        # Reshape img to (256, 256 * factor)
        # opencv format: (width, height)
        dim = (int(256 * factor), 256)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Get window of 256x256
        mid_idx = int(resized.shape[1] // 2)
        resized_f = resized[:, mid_idx-128:mid_idx+128]
    elif nb_cols < nb_rows:
        factor = nb_rows / nb_cols

        dim = (256, int(factor * 256))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        mid_idx = int(resized.shape[0] // 2)
        resized_f = resized[mid_idx-128:mid_idx+128]
    else:
        # reshape to 256x256
        resized_f = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    return resized_f

def process_imgs(imgs_path, save_path):
    for idx, img_path in enumerate(imgs_path):
        try:
            print("{} Process: {}".format(idx, img_path))

            # Read the image
            img = read_img(img_path)

            # resize properly the image
            resized_img = resize_img(img)

            # save the image
            img_name = img_path.split('/')[-1]
            save_path_img = os.path.join(save_path, img_name)
            save_img(save_path_img, resized_img)
        except Exception as e:
            print("Could not process image: {}".format(img_path))

def main(argv):
    input_folder, output_folder = argv[1], argv[2]

    # Create output folder and get images path
    create_folder(output_folder)
    img_list = glob.glob(os.path.join(input_folder, '*'))

    # Process all the images
    process_imgs(img_list, output_folder)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 reshape_img.py <input_folder> <output_folder>")
        os._exit(1)

    main(sys.argv)