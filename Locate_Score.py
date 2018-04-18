# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import skimage
import pylab
import os
import numpy as np
import cv2
from skimage.filters import threshold_otsu
import sys
import copy

redLower = np.array([170, 100, 100])
redUpper = np.array([179, 255, 255])

if __name__ == '__main__':
    all_list = os.listdir('./img_7_Red/')
    for k in range(len(all_list)):
        Img_RGB = cv2.imread(('./img_7_Red/%s' % all_list[k]).replace('//', ''))
        height,width,channel=Img_RGB.shape
        Img_RGB=cv2.resize(Img_RGB,(int(200*(width/height)),200))
        HSV = cv2.cvtColor(Img_RGB, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)
        LowerBlue = np.array([-10, 43, 46])
        UpperBlue = np.array([10, 255, 255])
        mask = cv2.inRange(HSV, LowerBlue, UpperBlue)
        RedThings = cv2.bitwise_and(Img_RGB, Img_RGB, mask=mask)
        cv2.imshow('fifi',RedThings)
        cv2.waitKey(0)
