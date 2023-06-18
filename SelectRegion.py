import cv2
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import os

# generate distortion field 
def generate_distortion_field(img, distortion):
    # generate distortion field
    distortion_field = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distortion_field[i, j] = distortion * np.random.normal(0, 1, 1)
    return distortion_field

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def main():
    folder = ''