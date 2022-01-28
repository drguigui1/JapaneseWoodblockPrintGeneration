# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import glob
import os

from pathlib import Path

WIDTH, HEIGHT = 256, 256

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):  
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def load_model(prototxt, caffemodel):
    net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
    cv.dnn_registerLayer('Crop', CropLayer)
    return net

def load_img(path):
    return cv.imread(path)

def save_img(path, img):
    cv.imwrite(path, img)

def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def make_pred(net):
    out = net.forward().squeeze()
    out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
    out = 255 * out
    return out.astype(np.uint8)

def process_image(img_path, net):
    image = load_img(img_path)

    inp = cv.dnn.blobFromImage(image, scalefactor=2.0, size=(WIDTH, HEIGHT),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
    net.setInput(inp)
    return make_pred(net)

def process_images(img_path, save_path, net):
    folders = glob.glob(os.path.join(img_path, '*'))

    idx = 0
    for folder_path in folders:
        files = glob.glob(os.path.join(folder_path, '*'))

        # Create artist folder
        folder_name = folder_path.split('/')[-1]
        save_path_folder = os.path.join(save_path, folder_name)
        create_folder(save_path_folder)

        for img_path in files:
            print("{} -- {}".format(idx + 1, img_path))

            # Process one image (Load / Prediction)
            pred = process_image(img_path, net)

            # Save the prediction
            img_name = img_path.split('/')[-1]
            save_path_img = os.path.join(save_path_folder, img_name)
            save_img(save_path_img, pred)
    
            idx += 1

def main():
    prototxt_path = 'src/deploy.prototxt'
    caffemodel_path = 'src/hed_pretrained_bsds.caffemodel'
    save_path = 'hed_preds/'
    img_path = 'reshaped_data/'

    # Create folder to save data if not exist
    create_folder(save_path)

    # Load the model and make prediction on all images
    net = load_model(prototxt_path, caffemodel_path)
    process_images(img_path, save_path, net)

if __name__ == "__main__":
    main()