# -*- coding: utf-8 -*-
import numpy as np
import skimage.measure
import os
import cv2
import Consecutive_dirp

def f_CreateImg(Img,no,i,x0,y0,x1,y1,filename_no_suffix):
    columns = y1-y0
    rows = x1-x0
    #print rows,columns
    NewImage = np.ones((rows,columns), np.uint8)
    NewImage=NewImage*255
    for x in range(rows):
        for y in range(columns):
            if(Img[x+x0][y+y0]==i):
                NewImage[x][y]=0
    #cv2.imshow("NewImage", NewImage)
    # #cv2.waitKey(0)
    if(no!=-1):
        cv2.imwrite('./img_Split/%s_%i.jpg' % (filename_no_suffix,no), NewImage)
    else:              #未成功分离
        cv2.imwrite('./img/%s.jpg' % filename_no_suffix, NewImage)

def f_connectivity(filename):
    img = cv2.imread(filename.replace('//', ''))
    filename=filename.split('/')[-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, columns = img.shape
    ret, th = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)         #白色1   黑色0
    #print "%s该图像阈值为：%i" % (filename,ret)
    con,num=skimage.measure.label(th, connectivity=2, background=1, return_num=True)
    #con中，0是背景，1，2，3等是各个连通区域
    count_all=[]
    for i in range(0,num):        #计算每个连通区域的点数目,如果小于指定阈值，则视为噪点
        count=0
        for x in range(rows):
            for y in range(columns):
                if(con[x][y] == i + 1):
                    count+=1
        if(count<=0.005*rows*columns):              #噪点阈值为0.005！！！
            for x in range(rows):
                for y in range(columns):
                    if (con[x][y] == i + 1):
                        con[x][y]=0
    con, num = skimage.measure.label(con, connectivity=2, background=0, return_num=True)
    for i in range(0,num):        #计算每个连通区域的点数目,如果小于指定阈值，则视为噪点
        count=0
        for x in range(rows):
            for y in range(columns):
                if(con[x][y] == i + 1):
                    count+=1
        count_all.append(count)

    gravity_cole = []
    rows_array=[]
    columns_array=[]
######################找出行列的最大数和最小数#######################
    for i in range(1,len(count_all)+1):
        columns_margin = []
        rows_margin = []
        flagl = 1;flagr = 1
        gravity_point=0
        for y in range(columns):                    ########两边往当中找########
            for x in range(rows):
                if int(con[x][y]) == i:
                    gravity_point+=y                ########用来判定数字从左到右的顺序############
                if (flagl and int(con[x][y]) == i):
                    columns_margin.append(y)
                    flagl=0
                if (flagr and int(con[x][columns-y-1]) == i):
                    columns_margin.append(columns-y-1)
                    flagr=0
        #print columns_margin
        flagt=1;flagb=1
        for x in range(rows):                       ########上下往当中找########
            for y in range(columns):
                if (flagt and int(con[x][y]) == i):
                    rows_margin.append(x)
                    flagt=0
                if (flagb and int(con[rows-x-1][y]) == i):
                    rows_margin.append(rows-x-1)
                    flagb=0
        #print rows_margin
        gravity_cole.append((gravity_point,i))
        rows_array.append(rows_margin)
        columns_array.append(columns_margin)

    if(len(gravity_cole)==2):
        if gravity_cole[0][0] > gravity_cole[1][0]:  # 两个连通区域只有两个比较元素  #no,con中i的顺序
            f_CreateImg(con, 1, 2, min(rows_array[1]), min(columns_array[1]), max(rows_array[1]),max(columns_array[1]), filename.split('.')[0])
            f_CreateImg(con, 2, 1, min(rows_array[0]), min(columns_array[0]), max(rows_array[0]), max(columns_array[0]),filename.split('.')[0])
        else:
            f_CreateImg(con, 1, 1, min(rows_array[0]), min(columns_array[0]), max(rows_array[0]),max(columns_array[0]), filename.split('.')[0])
            f_CreateImg(con, 2, 2, min(rows_array[1]), min(columns_array[1]), max(rows_array[1]),max(columns_array[1]), filename.split('.')[0])
        return 0
    elif(len(gravity_cole)==1):         #未成功分离
        f_CreateImg(con, -1, 1, min(rows_array[0]), min(columns_array[0]), max(rows_array[0]), max(columns_array[0]),filename.split('.')[0])
        return 1

def Clean_File():
    File=os.listdir('./img_Split/')
    m=len(File)
    for i in range(m):
        try:
            os.remove('./img_Split/%s' % File[i])
        except Exception as e:
            print('Has removed')

    File=os.listdir('./img/')
    m=len(File)
    for i in range(m):
        try:
            os.remove('./img/%s' % File[i])
        except Exception as e:
            print('Has removed')


if __name__ == '__main__':
    Clean_File()
    all_list=os.listdir('./img_Raw/')
    for i in range(len(all_list)):
        if(f_connectivity(('./img_Raw/%s' %all_list[i]))):
            print ("%s该图像未被分割" % all_list[i])
            Consecutive_dirp.drip_it('%s' % all_list[i])
        else:
            print ("%s该图像已被分割" % all_list[i])

    all_list1=os.listdir('./img/')
    for i in range(len(all_list1)):
        f_connectivity('./img/%s' %all_list1[i])         #对水滴算法后的图片进行连通区域分析



