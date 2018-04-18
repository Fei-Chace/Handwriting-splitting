# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np

def main():
    all_list = os.listdir('./img_4_Split/')
    for i in range(len(all_list)):
        Label_list = os.listdir('./img_5_Label/')
        Img = cv2.imread(('./img_4_Split/%s' % all_list[i]).replace('//', ''))
        print('*****************%r' %('./img_4_Split/%s' % all_list[i]).replace('//', ''))
        number=int(all_list[i].split('_')[0])
        first=int(number/10)
        second=int(number%10)
        print('%d %d' %(first,second))
        print(all_list[i].split('_')[1].split('.')[0])
        if int(all_list[i].split('_')[1])==1:
            filename = ('%s_0.jpg' % first)
            count=1
            while(filename in Label_list):
                filename=('%s_%d.jpg' %(first,count))
                count+=1
            print(filename)
            cv2.imwrite('./img_5_Label/%s' % filename, Img)

        if int(all_list[i].split('_')[1])==2:
            filename = ('%s_0.jpg' % second)
            count=1
            while(filename in Label_list):
                filename=('%s_%d.jpg' %(second,count))
                count+=1
            print(filename)
            cv2.imwrite('./img_5_Label/%s' % filename, Img)

if __name__ == '__main__':
    main()