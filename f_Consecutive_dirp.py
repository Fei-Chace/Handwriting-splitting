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

Re_Size = 160
Drip_Num = Re_Size

def find_margin(img, rows, columns):  # 找到字符的边界
    flagl = 1;
    flagr = 1
    for x in range(rows):
        for y in range(columns):
            if (flagl and img[x][y] == 0):
                row_left = x
                flagl = 0
            if (flagr and img[rows - 1 - x][y] == 0):
                row_right = rows - 1 - x
                flagr = 0
            if (not flagr and not flagl):
                break
    flagt = 1;
    flagd = 1
    for y in range(columns):
        for x in range(rows):
            if (flagt and img[x][y] == 0):
                col_top = y
                flagt = 0
            if (flagd and img[x][columns - 1 - y] == 0):
                col_down = columns - 1 - y
                flagd = 0
            if (not flagt and not flagd):
                break
    return row_left, col_top, row_right, col_down

def Img_Create(img, filename_have_suffix):
    rows, columns = img.shape  # shape返回的是(高度， 宽度) = (y , x)
    size = (Re_Size, int(rows * Re_Size * 1.0 / columns))
    Img_Resize = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    Otsu_Threshold = threshold_otsu(Img_Resize)
    for y in range(0, size[0]):
        for x in range(0, size[1]):
            if (Img_Resize[x][y] > Otsu_Threshold):
                Img_Resize[x][y] = 255
            else:
                Img_Resize[x][y] = 0
    cv2.imwrite('./img_2_Resize/%s.jpg' % filename_have_suffix.split('.')[0], Img_Resize)
    return Img_Resize  # 将float形数组转换为int

def Drop_it_Bottom_Top(img):
    height, width = img.shape  # shape返回的是(高度， 宽度) = (y , x)
    count = 0
    while count < Drip_Num:
        for t in range(width):  # 从最左到最右依次洒水
            row = height - 1;
            col = t  # 对滴水位置的初始化
            check_point = []
            momentum = 1  # 水滴动能，默认向右
            while True:  # 开始水滴的运动轨迹
                if img[row][col] == 0 or img[row][
                    col] == 200 or row == 0:  # 如果发现当前点是着黑色直接滴下一滴水 或者 已经滴到图片的最底部(不太可能发生)
                    count += 1
                    break
                if check_point.count((row, col)) >= 3:  # 即发现了来回滚动现象
                    img[row][col] = 200
                    count += 1
                    break
                check_point.append((row, col))
                if row >= 0 and img[row - 1][col] == 255:  # 开始确定水滴下一步路径        #往下
                    row = row - 1
                elif momentum == 0 and col - 1 >= 0 and img[row][col - 1] == 255:  # 动能左，并左移
                    col = col - 1
                elif momentum == 1 and col + 1 <= width - 1 and img[row][col + 1] == 255:  # 动能右，并右移
                    col = col + 1
                elif momentum == 1 and col - 1 >= 0 and img[row][col - 1] == 255:  # 动能右，左移
                    col = col - 1
                    momentum = 0  # 重新定义水滴动能
                elif momentum == 0 and col + 1 <= width - 1 and img[row][col + 1] == 255:  # 动能左，右移
                    col = col + 1
                    momentum = 1  # 重新定义水滴动能
                elif ((row - 1 >= 0 and (
                        img[row - 1][col] == 0 or img[row - 1][col] == 200)) or row - 1 < 0) \
                        and ((col - 1 >= 0 and (img[row][col - 1] == 0 or img[row][col - 1] == 200)) or col - 1 < 0) \
                        and ((col + 1 <= width - 1 and (img[row][col + 1] == 0 or img[row][
                    col + 1] == 127)) or col + 1 > width - 1):  # 发现要么卡住了，要么贴边线，要么结束
                    img[row][col] = 200
                    count += 1
                    break
                ##############################此处为进度条##############################
                rate = (count / Drip_Num) * 0.5 + 0.5  # count读完的百分比,进图条,后一半
                count_rate = round(rate * 100)
                r = '\r[%s]%d%%' % ("=" * count_rate, count_rate,)
                sys.stdout.write(r)
                sys.stdout.flush()
    print('\n')
    #################################开始去除非局部最小值#################################
    for x in range(height):
        for y in range(width):  # 开始对下方是黑色的底部区域进行标记
            if img[x][y] == 200 and (x - 1 < 0 or img[x - 1][y] == 0 or img[x - 1][y] == 255):
                img[x][y] = 100
        for y in range(width):  # 排除非局部最小值
            if img[x][y] == 100:
                flagl = flagr = 1  # 左右边移动的游标
                cursor = 1
                while True:
                    if flagl and (y - cursor < 0 or img[x][y - cursor] == 0 or img[x][y - cursor] == 255):  # 发现贴边
                        flagl = 0
                    elif flagl and img[x][y - cursor] == 200:  # 发现是假的极值
                        img[x][y] = 200
                        flagl = 0
                    if flagr and (y + cursor >= width or img[x][y + cursor] == 0 or img[x][y + cursor] == 255):  # 发现贴边
                        flagr = 0
                    elif flagr and img[x][y + cursor] == 200:  # 发现是假的极值
                        img[x][y] = 200
                        flagr = 0
                    if flagl == 0 and flagr == 0:
                        break
                    cursor += 1

        dy = 0  # 排除贴在边缘的最小值,重置为200
        while img[x][dy] == 100:
            img[x][dy] = 200
            dy += 1
        dy = width - 1
        while img[x][dy] == 100:
            img[x][dy] = 200
            dy -= 1

    return img

def Drop_it_Top_Bottom(img):
    height, width = img.shape  # shape返回的是(高度， 宽度) = (y , x)
    count = 0
    while count < Drip_Num:
        for t in range(width):  # 从最左到最右依次洒水
            row = 0;
            col = t  # 对滴水位置的初始化
            check_point = []  # 记录路径经过几次
            momentum = 0  # 水滴动能，默认向左
            while True:  # 开始水滴的运动轨迹
                if row == height - 1 or img[row][col] == 0 or img[row][
                    col] == 200:  # 如果发现当前点是着黑色直接滴下一滴水 或者 已经滴到图片的最底部(不太可能发生)
                    count += 1
                    break
                if check_point.count((row, col)) >= 3:  # 即发现了来回滚动现象
                    img[row][col] = 200
                    count += 1
                    break
                check_point.append((row, col))
                if row + 1 <= height - 1 and img[row + 1][col] == 255:  # 开始确定水滴下一步路径        #往下
                    row = row + 1
                elif momentum == 0 and col - 1 >= 0 and row + 1 <= height - 1 and img[row + 1][
                    col - 1] == 255:  # 动能左，并左下移
                    col -= 1
                    row += 1
                elif momentum == 1 and col + 1 <= width - 1 and row + 1 <= height - 1 and img[row + 1][
                    col + 1] == 255:  # 动能右，并右下移
                    col += 1
                    row += 1
                elif momentum == 1 and col - 1 >= 0 and row + 1 <= height - 1 and img[row + 1][
                    col - 1] == 255:  # 动能左，并左下移
                    col -= 1
                    row += 1
                    momentum = 0
                elif momentum == 0 and col + 1 <= width - 1 and row + 1 <= height - 1 and img[row + 1][
                    col + 1] == 255:  # 动能右，并右下移
                    col += 1
                    row += 1
                    momentum = 1
                elif momentum == 0 and col - 1 >= 0 and img[row][col - 1] == 255:  # 动能左，左下不能，只能左移
                    col -= 1
                elif momentum == 1 and col + 1 <= width - 1 and img[row][col + 1] == 255:  # 动能右，右下不能，只能右移
                    col += 1
                elif momentum == 1 and col - 1 >= 0 and img[row][col - 1] == 255:  # 动能左，左下不能，只能左移
                    col -= 1
                    momentum = 0
                elif momentum == 0 and col + 1 <= width - 1 and img[row][col + 1] == 255:  # 动能右，右下不能，只能右移
                    col += 1
                    momentum = 1

                elif ((row + 1 <= height - 1 and col - 1 >= 0 and (img[row + 1][col - 1] == 0 or img[row + 1][
                    col - 1] == 200)) or row + 1 > height - 1 or col - 1 < 0) and \
                        ((row + 1 <= height - 1 and col + 1 <= width - 1 and (
                                img[row + 1][col + 1] == 0 or img[row + 1][
                            col + 1] == 200)) or row + 1 > height - 1 or col + 1 > width - 1) and \
                        ((row + 1 <= height - 1 and (
                                img[row + 1][col] == 0 or img[row + 1][col] == 200)) or row + 1 > height - 1) \
                        and ((col - 1 >= 0 and (img[row][col - 1] == 0 or img[row][col - 1] == 200)) or col - 1 < 0) \
                        and ((col + 1 <= width - 1 and (img[row][col + 1] == 0 or img[row][
                    col + 1] == 200)) or col + 1 > width - 1):  # 发现要么卡住了，要么贴边线，要么结束
                    img[row][col] = 200
                    count += 1
                    break

                ##############################此处为进度条##############################
                rate = (count / Drip_Num) * 0.5  # count读完的百分比,进图条,前一半
                count_rate = round(rate * 100)
                r = '\r[%s]%d%%' % ("=" * count_rate, count_rate,)
                sys.stdout.write(r)
                sys.stdout.flush()

    for x in range(height):
        for y in range(width):  # 开始对下方是黑色的底部区域进行标记
            if img[x][y] == 200 and (x + 1 >= height or img[x + 1][y] == 0 or img[x + 1][y] == 255):
                img[x][y] = 100
        for y in range(width):  # 排除非局部最小值
            if img[x][y] == 100:
                flagl = flagr = 1  # 左右边移动的游标
                cursor = 1
                while True:
                    if flagl and (y - cursor < 0 or img[x][y - cursor] == 0 or img[x][y - cursor] == 255):  # 发现贴边
                        flagl = 0
                    elif flagl and img[x][y - cursor] == 200:  # 发现是假的极值
                        img[x][y] = 200
                        flagl = 0
                    if flagr and (y + cursor >= width or img[x][y + cursor] == 0 or img[x][y + cursor] == 255):  # 发现贴边
                        flagr = 0
                    elif flagr and img[x][y + cursor] == 200:  # 发现是假的极值
                        img[x][y] = 200
                        flagr = 0
                    if flagl == 0 and flagr == 0:
                        break
                    cursor += 1

        dy = 0  # 排除贴在边缘的最小值,重置为200
        while img[x][dy] == 100:
            img[x][dy] = 200
            dy += 1
        dy = width - 1
        while img[x][dy] == 100:
            img[x][dy] = 200
            dy -= 1

    return img

def Four_Connectivity(img, x, y, t, check_img, type):
    height, width = img.shape
    #if img[x][y] == 100:                #暂时为局部最小点
        #img[x][y] = 200
    check_img[x][y] = 0  # 标记是否count过
    count = 1  # 第一个初始count
    if (t < 3):  # 向上迭代小于等于三次
        if not type and x - 1 >= 0 and check_img[x - 1][y] and (
                img[x - 1][y] == 200 or img[x - 1][y] == 100):  # type==0表示从上往下滴水
            count += Four_Connectivity(img, x - 1, y, t + 1, check_img, type)
        if type and x + 1 < height and check_img[x + 1][y] and (
                img[x + 1][y] == 200 or img[x + 1][y] == 100):  # type==1表示从下往上滴水
            count += Four_Connectivity(img, x + 1, y, t + 1, check_img, type)
        # if x+1<height and check_img[x+1][y] and (img[x+1][y]==200 or img[x+1][y]==100):
        # count+=Four_Connectivity(img,x+1,y,t-1,check_img)
        if y - 1 >= 0 and check_img[x][y - 1] and (img[x][y - 1] == 200 or img[x][y - 1] == 100):  # 往两边检索
            count += Four_Connectivity(img, x, y - 1, t, check_img, type)
        if y + 1 < width and check_img[x][y + 1] and (img[x][y + 1] == 200 or img[x][y + 1] == 100):
            count += Four_Connectivity(img, x, y + 1, t, check_img, type)  # 只有向上的时候需要t+1
    return count

def Search_Split_Point(img, type):  # 开始寻找角度最小的凹陷点
    height, width = img.shape  # shape返回的是(高度， 宽度) = (y , x)
    count_sum = []
    for x in range(height):
        for y in range(width):
            if img[x][y] == 100:
                check_img = np.ones((height, width))
                count = Four_Connectivity(img, x, y, 1, check_img, type)  # 寻找四连通区域,向上一次
                if (count > 2):  # 可能会存在无效点
                    count_sum.append((x, y, count))
    return count_sum

def Plot_Line(x0, y0, x1, y1, filename, Gray_img, is_show):  # DDA进行简单的画直线切分示意
    L = max(abs(x1 - x0), abs(y1 - y0))  # 开始进行dda算法
    dx = (x1 - x0) * 1.0 / L;
    dy = (y1 - y0) * 1.0 / L
    x = x0;
    y = y0
    for i in range(0, L):
        # if(int(round(x))>=height):      x -= 1             #防止越界，一般不可能
        # if (int(round(y)) >= width):    y -= 1             #防止越界，一般不可能
        Gray_img[int(round(x))][int(round(y))] = 255
        x = x + dx;
        y = y + dy
    Connect_img, num = skimage.measure.label(Gray_img, connectivity=1, background=255, return_num=True)  # 返回的已经是连通区域图像
    if (is_show):
        cv2.imwrite('./img_3_plot/%s' % filename, Gray_img)
        #cv2.imshow('%s' % filename, Gray_img)
        #cv2.waitKey(0)
        return Connect_img, Gray_img, num  # Gray_img即Split_img
    return num

def Scan_Line(x0, y0, x1, y1, filename, Gray_img, type):  # (x0,y0)上滴点,(x1,y1)下滴点
    height, width = Gray_img.shape
    best_tpxy = []  # 最优点
    candidate_tpxy = []  # 候选点
    tpxy = []
    if (type):
        x2 = x1 + 1;
        y2 = y1  # 寻找垂直点
        while (x2 > 0 and not Gray_img[x2][y2] == 255):
            x2 += 1
    else:
        x2 = x1 - 1;
        y2 = y1  # 寻找垂直点
        while (x2 > 0 and not Gray_img[x2][y2] == 255):
            x2 -= 1

    L_line = max(abs(x2 - x0), abs(y2 - y0))
    dline_x = (x2 - x0) * 1.0 / L_line;
    dline_y = (y2 - y0) * 1.0 / L_line
    tpp_x = x0;
    tpp_y = y0  # 临时连线上的所有点依次遍历
    count_min = float("inf")  # 赋值无限大

    for i in range(0, L_line):
        tpp_x = tpp_x + dline_x;
        tpp_y = tpp_y + dline_y
        tp_x = round(tpp_x);
        tp_y = round(tpp_y)  # 临时点
        L = max(abs(x1 - tp_x), abs(y1 - tp_y))
        if (y1 - tp_y != 0 and abs(x1 - tp_x) / abs(y1 - tp_y) < 1):  # 表示这条线的斜率
            gradient = abs(x1 - tp_x) / abs(y1 - tp_y)
            coefficient = 10 * (1 - gradient)  # 如果线比较水平，为了防止贴边错识别，需引入一个系数
        else:
            coefficient = 0

        dx = (tp_x - x1) * 1.0 / L;
        dy = (tp_y - y1) * 1.0 / L
        x = x1 * 1.0;
        y = y1 * 1.0
        count = 0;
        check = 0  # check用来检查是否已经画出边界
        flag = 1
        while (int(round(x)) > 0 and int(round(x)) < height - 1 and int(round(y)) > 0 and int(round(y)) < width - 1):
            if (flag and Gray_img[int(round(x))][int(round(y))] == 0):  # Let`s Go!
                flag = 0
            elif (not flag and check >= 1 and count > coefficient and Gray_img[int(round(x))][
                int(round(y))] == 255):  # 容忍度为1,此个count为人为设定！
                break
            elif (not flag and Gray_img[int(round(x))][int(round(y))] == 0):
                count += 1
                if (check > 0):
                    check -= 1
            elif (not flag and Gray_img[int(round(x))][int(round(y))] == 255):
                check += 1
            x = x + dx;
            y = y + dy
        if (count <= count_min):
            count_min = count
            try:
                tpxy.pop()
            except:
                pass
            tpxy.append((round(x), round(y), count))
        else:  # 存入局部最小值
            if ((tpxy[0][0], tpxy[0][1], tpxy[0][2]) not in candidate_tpxy):
                candidate_tpxy.append((tpxy[0][0], tpxy[0][1], tpxy[0][2]))

    if (type):
        print("候选上滴分割点集%r" % candidate_tpxy)
    else:
        print("候选下滴分割点集%r" % candidate_tpxy)

    count_min = float("inf")  # 赋值无限大
    for k in range(len(candidate_tpxy)):
        num = Plot_Line(x1, y1, candidate_tpxy[k][0], candidate_tpxy[k][1], filename, copy.deepcopy(Gray_img), 0)
        if (num == 2 and candidate_tpxy[k][2] <= count_min):
            count_min = candidate_tpxy[k][2]
            try:
                best_tpxy.pop()
            except:
                pass
            best_tpxy.append((candidate_tpxy[k][0], candidate_tpxy[k][1], candidate_tpxy[k][2]))
        elif (num == 1):
            continue
    if (len(best_tpxy)):
        Plot_Line(x1, y1, best_tpxy[0][0], best_tpxy[0][1], filename, copy.deepcopy(Gray_img), 0)
        return x1, y1, best_tpxy[0][0], best_tpxy[0][1]
    else:
        print("拒绝识别！")
        return 0, 0, 0, 0

def Choose_Split_Point(filename, Gray_img, top_count, down_count):
    height, width = Gray_img.shape  # shape返回的是(高度， 宽度) = (y , x)
    ans = []
    for i in range(0, len(top_count)):
        for j in range(0, len(down_count)):
            if down_count[j][1] > top_count[i][1]:  # down的点一定要在top的右边
                print("###############%r %r" % (top_count[i], down_count[j]))
                x0 = top_count[i][0];
                y0 = top_count[i][1];
                z0 = top_count[i][2]
                x1 = down_count[j][0];
                y1 = down_count[j][1];
                z1 = down_count[j][2]
                L = max(abs(x1 - x0), abs(y1 - y0))  # 开始进行dda算法
                dx = (x1 - x0) * 1.0 / L;
                dy = (y1 - y0) * 1.0 / L
                x = x0 * 1.0;
                y = y0 * 1.0
                sum_count = 0  # 所有的DDA点
                count = 0  # 所有有效点
                for k in range(0, L):
                    if Gray_img[int(round(x))][int(round(y))] == 0:  # 发现是着色点
                        count += 1
                    sum_count += 1
                    x = x + dx;
                    y = y + dy
                print('重合比 %i/%i' % (count, sum_count))  # DDA重合数和DDA总数
                # print('###%r%r'%(top_count[i][1],down_count[j][1]))
                if count * 1.0 / sum_count >= 0.7:
                    ans.append(((x0, y0, z0), (x1, y1, z1)))  # 变成一个三维数

    min = float("inf")  # count最大值
    Best_Pair = []
    count_P = []
    for i in range(0, len(ans)):
        ################惩罚值P=[2*|x-x0|/height]*[2*|y-y0|/width]}################
        """
        x=(ans[i][0][0]+ans[i][1][0])/2;y=(ans[i][0][1]+ans[i][1][1])/2
        P=pow(abs(x-x0)/height+1)*(abs(y-y0)/width+1)"""
        x0 = height / 2;
        y0 = width / 2
        P = (abs(x0 - ans[i][0][0]) / x0) * (abs(y0 - ans[i][0][1]) / y0) * (abs(x0 - ans[i][1][0]) / x0) * (
                abs(y0 - ans[i][1][1]) / y0)
        print("%r %r %r %r %r" % (ans[i][0][0], ans[i][0][1], ans[i][1][0], ans[i][1][1], P))
        ###########################################################################
        count_P.append((ans[i][0][2] + ans[i][1][2], P, (ans[i][0][2] + ans[i][1][2]) * P))
        if (ans[i][0][2] + ans[i][1][2]) * P <= min:
            try:
                Best_Pair.pop()  # 第一次不可能为空，所以要try
            except:
                pass
            Best_Pair.append(ans[i])
            min = (ans[i][0][2] + ans[i][1][2]) * P
    if len(Best_Pair) == 0:
        print('未找到合适上下点对，该字符拒识！')
        return;
    print("分割线与中心点的偏离率 %r" % count_P)
    x0 = Best_Pair[0][0][0];
    y0 = Best_Pair[0][0][1]  # 从上而下的水滴
    x1 = Best_Pair[0][1][0];
    y1 = Best_Pair[0][1][1]  # 从下而上的水滴
    Plot_Line(x0, y0, x1, y1, filename, copy.deepcopy(Gray_img), 0)
    px0, py0, px1, py1 = Scan_Line(x1, y1, x0, y0, filename, copy.deepcopy(Gray_img), 1)  # 扫描切分上滴点
    px2, py2, px3, py3 = Scan_Line(x0, y0, x1, y1, filename, copy.deepcopy(Gray_img), 0)  # 扫描切分下滴点
    print("两条分割线的端点 %r %r %r %r %r %r %r %r" % (px0, py0, px1, py1, px2, py2, px3, py3))
    if (px0 and py0 and px1 and py1 and px2 and py2 and px3 and py3):
        Plot_Line(px0, py0, px1, py1, filename, Gray_img, 0)  # 用DDA绘制该切分线
        Plot_img = Plot_Line(px2, py2, px3, py3, filename, Gray_img, 1)  # 画好切分线的图像
        return Plot_img
    else:
        print("由于未找到分割线的端点，无法绘制分割直线！")

def Remove_Redundant_Writing(filename, Connect_Img, Split_Img, num):
    filename=filename[0:2]
    height, width = Connect_Img.shape  # shape返回的是(高度， 宽度) = (y , x)
    count={}
    pos={}
    for x in range(height):
        for y in range(width):
            if (Connect_Img[x][y] != 0):  # 不需要计算0的个数
                if (Connect_Img[x][y] not in count.keys()):
                    count[Connect_Img[x][y]] = [1,y]
                else:
                    count[Connect_Img[x][y]][0] += 1            #count[][0]为数量，count[][1]为y累积和
                    count[Connect_Img[x][y]][1] += y
    for key in count:
        pos[key]=count[key][1]/count[key][0]
        print("++++++++++++++++%r" %pos[key])
    left_num=min(zip(pos.values(), pos.keys()))
    right_num=max(zip(pos.values(), pos.keys()))
    print('************************%r*******************%r'%(left_num,right_num))
    left_array=np.ones((height,width),dtype=np.int)
    right_array = np.ones((height, width),dtype=np.int)
    left_array=left_array*255
    right_array=right_array*255
    for x in range(height):
        for y in range(width):
            if(Connect_Img[x][y]==left_num[1]):             #为什么要left_num[0]????
                left_array[x][y]=0
            elif(Connect_Img[x][y]==right_num[1]):
                right_array[x][y]=0

    count=0
    Split_list = os.listdir('./img_4_Split/')
    while('%s_1_%d.jpg' % (filename,count) in Split_list):
        count+=1
    cv2.imwrite('./img_4_Split/%s_1_%d.jpg' % (filename,count), left_array)
    cv2.imwrite('./img_4_Split/%s_2_%d.jpg' % (filename,count), right_array)

def main():
    all_list = os.listdir('./img_1_Raw/')
    for k in range(len(all_list)):
        Img_Original = cv2.imread(('./img_1_Raw/%s' % all_list[k]).replace('//', ''))
        print('#%s' % all_list[k])
        try:
            Img_Gray = cv2.cvtColor(Img_Original, cv2.COLOR_BGR2GRAY)  # 将彩色图片变为灰度图片
        except:
            Img_Gray = Img_Original
        rows, columns = Img_Gray.shape  # shape返回的是(高度， 宽度) = (y , x)w
        Otsu_Threshold = threshold_otsu(Img_Gray)  # 对其进行二值化
        for x in range(rows):
            for y in range(columns):
                if (Img_Gray[x][y] > Otsu_Threshold):
                    Img_Gray[x][y] = 255
                else:
                    Img_Gray[x][y] = 0
        row_left, col_top, row_right, col_down = find_margin(Img_Gray, rows, columns)  # 寻找外界矩形边界
        Img_Crop = Img_Gray[row_left:row_right + 1, col_top:col_down + 1]
        Resize_Img = Img_Create(Img_Crop, all_list[k])  # 在Resize文档中创建该图像

        Img_TB = Drop_it_Top_Bottom(copy.deepcopy(Resize_Img))  # 从上向下滴水
        top_count = Search_Split_Point(Img_TB, 0)  # 寻找最优的最小值

        Img_BT = Drop_it_Bottom_Top(copy.deepcopy(Resize_Img))  # 从下往上滴水
        down_count = Search_Split_Point(Img_BT, 1)  # 寻找最优的最小值

        # top_count.sort(key=lambda x: x[1])                 #按照第二个元素col从小到大排序
        # down_count.sort(key=lambda x: x[1])                #按照第二个元素col从小到大排序

        print("上滴点的坐标(x,y，count) %r" % top_count);
        print("下滴点的坐标(x,y，count) %r" % down_count)
        ###############################切分点显示###############################

        Img_Display = cv2.cvtColor(Resize_Img, cv2.COLOR_GRAY2BGR)
        for i in range(len(top_count)):
            Img_Display[top_count[i][0]][top_count[i][1]][2] = 255
            Img_Display[top_count[i][0]][top_count[i][1]][1] = 0
            Img_Display[top_count[i][0]][top_count[i][1]][0] = 0

            Img_Display[top_count[i][0] - 1][top_count[i][1]][2] = 255
            Img_Display[top_count[i][0] - 1][top_count[i][1]][1] = 0
            Img_Display[top_count[i][0] - 1][top_count[i][1]][0] = 0

            Img_Display[top_count[i][0] + 1][top_count[i][1]][2] = 255
            Img_Display[top_count[i][0] + 1][top_count[i][1]][1] = 0
            Img_Display[top_count[i][0] + 1][top_count[i][1]][0] = 0

            Img_Display[top_count[i][0]][top_count[i][1] - 1][2] = 255
            Img_Display[top_count[i][0]][top_count[i][1] - 1][1] = 0
            Img_Display[top_count[i][0]][top_count[i][1] - 1][0] = 0

            Img_Display[top_count[i][0]][top_count[i][1] + 1][2] = 255
            Img_Display[top_count[i][0]][top_count[i][1] + 1][1] = 0
            Img_Display[top_count[i][0]][top_count[i][1] + 1][0] = 0

        for i in range(len(down_count)):
            Img_Display[down_count[i][0]][down_count[i][1]][2] = 255
            Img_Display[down_count[i][0]][down_count[i][1]][1] = 0
            Img_Display[down_count[i][0]][down_count[i][1]][0] = 0

            Img_Display[down_count[i][0] - 1][down_count[i][1]][2] = 255
            Img_Display[down_count[i][0] - 1][down_count[i][1]][1] = 0
            Img_Display[down_count[i][0] - 1][down_count[i][1]][0] = 0

            Img_Display[down_count[i][0] + 1][down_count[i][1]][2] = 255
            Img_Display[down_count[i][0] + 1][down_count[i][1]][1] = 0
            Img_Display[down_count[i][0] + 1][down_count[i][1]][0] = 0

            Img_Display[down_count[i][0]][down_count[i][1] - 1][2] = 255
            Img_Display[down_count[i][0]][down_count[i][1] - 1][1] = 0
            Img_Display[down_count[i][0]][down_count[i][1] - 1][0] = 0

            Img_Display[down_count[i][0]][down_count[i][1] + 1][2] = 255
            Img_Display[down_count[i][0]][down_count[i][1] + 1][1] = 0
            Img_Display[down_count[i][0]][down_count[i][1] + 1][0] = 0

        #cv2.imshow('%s' % all_list[k], Img_Display)
        #cv2.waitKey(0)
        #######################################################################
        try:
            Connect_Img, Split_img, num = Choose_Split_Point(all_list[k], Resize_Img, top_count, down_count)
            if (num == 3):
                Remove_Redundant_Writing(all_list[k], Connect_Img, Split_img, num)  # num为连通区域的数量
        except:
            pass
        ####################################去除冗余笔画####################################

if __name__ == '__main__':
    main()

"""    for kx in range(0,height):
        for ky in range(0,width):
            if(kx==x1 and ky==y1):
                print(4, end=' ')
            elif (kx == candidate_tpxy[0][0] and ky == candidate_tpxy[0][1]):
                print(6, end=' ')
            elif(kx==x2 and ky==y2):
                print(9, end=' ')
            elif(Gray_img[kx][ky]==255):
                print(1, end=' ')
            else:
                print(0, end=' ')
        print('\n', end='')"""