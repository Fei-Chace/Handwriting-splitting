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

def main():
    all_list = os.listdir('./img_5_Label/')
    for i in range(len(all_list)):
        first = all_list[i].split('_')[0]
        Img = cv2.imread('./img_5_Label/%s' % all_list[i], 0)
        height, width = Img.shape
        Otsu_Threshold = threshold_otsu(Img)
        print(height, width)
        flagt = flagb = 1
        for x in range(height):
            for y in range(width):
                if(Img[x][y]>Otsu_Threshold):
                    Img[x][y]=1
                elif(Img[x][y]<=Otsu_Threshold):
                    Img[x][y] = 0
        for x in range(height):
            for y in range(width):
                if (flagt and Img[x][y] == 0):
                    top_line = x
                    flagt = 0
                if (flagb and Img[height - x - 1][y] == 0):
                    bottom_line = height - x - 1
                    flagb = 0
                if (not flagt and not flagb):
                    break
        flagl = flagr = 1
        for y in range(width):
            for x in range(height):
                if (flagl and Img[x][y] == 0):
                    left_line = y
                    flagl = 0
                if (flagr and Img[x][width - 1 - y] == 0):
                    right_line = width - y - 1
                    flagr = 0
                if (not flagl and not flagr):
                    break
        #print(top_line, bottom_line, left_line, right_line)
        dh = bottom_line + 1 - top_line
        dw = right_line + 1 - left_line
        Img2 = cv2.imread('./img_5_Label/%s' % all_list[i], 0)
        new_Img = Img2[top_line:bottom_line + 1, left_line:right_line + 1]
        if (dh > dw):
            ratio=dh *1.0/ 20
            dw = int(dw / ratio)
            dh = 20
        else:
            ratio = dw *1.0/ 20
            dh = int(dh/ratio)
            dw = 20
        print(dh,dw,ratio)
        new_Img = cv2.resize(new_Img, (dw,dh), interpolation=cv2.INTER_AREA)
        print(new_Img.shape)

        fin_Img=np.zeros((28,28))
        #fin_Img=fin_Img*255
        x_coe=int(round((28-dh)/2))
        y_coe=int(round((28-dw)/2))
        for x in range(0,dh):
            for y in range(0,dw):
                fin_Img[x+x_coe][y+y_coe]=255-new_Img[x][y]

        cv2.imwrite('./img_6_Normalization/%s/%s' % (first,all_list[i]), fin_Img)
        #cv2.imshow('fi', fin_Img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

