import os
import cv2
from skimage.filters import threshold_otsu
import sys
import copy

Re_Size=200
Drip_Num=Re_Size*3

def find_margin(img,rows,columns):
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
    return row_left,col_top,row_right,col_down

def Img_Create(img,filename_have_suffix):
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
    #cv2.imwrite('./img_2_Resize/%s.jpg' %filename_have_suffix.split('.')[0], Img_Resize)
    return Img_Resize             #将float形数组转换为int

def Drop_it_Top_Bottom(img):
    height, width = img.shape  # shape返回的是(高度， 宽度) = (y , x)
    count=0
    while count<Drip_Num:
        for t in range(width):      #从最左到最右依次洒水
            row=0;col=t                 #对滴水位置的初始化
            check_point=[]
            momentum=1      #水滴动能，默认向右
            while True:                 #开始水滴的运动轨迹
                if img[row][col]==0 or img[row][col]==200 or row==height-1:        #如果发现当前点是着黑色直接滴下一滴水 或者 已经滴到图片的最底部(不太可能发生)
                    count+=1
                    break
                check_point.append((row, col))
                if check_point.count((row, col))>=3:         #即发现了来回滚动现象
                    img[row][col]=200
                    count+=1
                    break
                if row+1<=height-1 and img[row+1][col]==255:    #开始确定水滴下一步路径        #往下
                    row=row+1
                elif momentum==0 and col-1>=0 and row+1<=height-1 and img[row+1][col-1]==255:         #动能左，并左下移
                    col-=1
                    row+=1
                elif momentum==1 and col+1<=width-1 and row+1<=height-1 and img[row+1][col+1]==255:   #动能右，并右下移
                    col+=1
                    row+=1
                elif momentum==1 and col-1>=0 and row+1<=height-1 and img[row+1][col-1]==255:         #动能左，并左下移
                    col-=1
                    row+=1
                    momentum= 0
                elif momentum==0 and col+1<=width-1 and row+1<=height-1 and img[row+1][col+1]==255:   #动能右，并右下移
                    col+=1
                    row+=1
                    momentum = 1

                elif momentum==0 and col-1>=0 and img[row][col-1]==255:         #动能左，左下不能，只能左移
                    col-=1
                elif momentum==1 and col+1<=width-1 and img[row][col+1]==255:   #动能右，右下不能，只能右移
                    col+=1
                elif momentum==1 and col-1>=0  and img[row][col-1]==255:         #动能左，左下不能，只能左移
                    col-=1
                    momentum = 0
                elif momentum==0 and col+1<=width-1 and img[row][col+1]==255:   #动能右，右下不能，只能右移
                    col+=1
                    momentum = 1
                elif ((row+1<=height-1 and col-1>=0 and (img[row+1][col-1]==0 or img[row+1][col-1]==200)) or row+1>height-1 or col-1<0) and \
                     ((row + 1 <= height - 1 and col + 1 <=width-1 and (img[row + 1][col + 1] == 0 or img[row + 1][col + 1] == 200)) or row + 1 > height - 1 or col + 1 >width-1) and\
                     ((row+1<=height-1 and (img[row+1][col]==0 or img[row+1][col]==200)) or row+1>height-1)\
                        and ((col-1>=0 and (img[row][col-1]==0 or img[row][col-1]==200)) or col-1<0)\
                        and ((col+1<=width-1 and (img[row][col+1]==0 or img[row][col+1]==200)) or col+1>width-1):   #发现要么卡住了，要么贴边线，要么结束
                    img[row][col] = 200
                    count+=1
                    break

                ##############################此处为进度条##############################
                rate = (count / Drip_Num)          # count读完的百分比,进图条,前一半
                count_rate = round(rate * 100)
                r = '\r[%s]%d%%' % ("=" * count_rate, count_rate,)
                sys.stdout.write(r)
                sys.stdout.flush()

    for x in range(height):
        for y in range(width):          #开始对下方是黑色的底部区域进行标记
            if img[x][y]==200 and(x+1>=height or img[x+1][y]==0 or img[x+1][y]==255):
                img[x][y]=100
        for y in range(width):          #排除非局部最小值
            if img[x][y] == 100:
                flagl=flagr=1             #左右边移动的游标
                cursor=1
                while True:
                    if flagl and (y-cursor<0 or img[x][y-cursor]==0 or img[x][y-cursor]==255):                      #发现贴边
                        flagl=0
                    elif flagl and img[x][y-cursor]==200:       #发现是假的极值
                        img[x][y]=200
                        flagl=0
                    if flagr and (y+cursor>=width or img[x][y+cursor]==0 or img[x][y+cursor]==255):                      #发现贴边
                        flagr=0
                    elif flagr and img[x][y+cursor]==200:       #发现是假的极值
                        img[x][y]=200
                        flagr=0
                    if flagl==0 and flagr==0:
                        break
                    cursor+=1

        dy=0                    #排除贴在边缘的最小值,重置为200
        while img[x][dy]==100:
            img[x][dy]=200
            dy+=1
        dy=width-1
        while img[x][dy]==100:
            img[x][dy]=200
            dy-=1

    return img

if __name__ == '__main__':
    #Img_Original = io.imread(('./img/%s' %filename).replace('//',''))
    all_list=os.listdir('./img_1_Raw/')
    for i in range(len(all_list)):
        Img_Original = cv2.imread(('./img_1_Raw/%s' % all_list[i]).replace('//', ''))
        print('####################%s' %all_list[i])
        try:
            Img_Gray = cv2.cvtColor(Img_Original, cv2.COLOR_BGR2GRAY)
        except:
            Img_Gray = Img_Original
        rows, columns = Img_Gray.shape  # shape返回的是(高度， 宽度) = (y , x)
        Otsu_Threshold = threshold_otsu(Img_Gray)           # 对其进行二值化
        for x in range(rows):
            for y in range(columns):
                if (Img_Gray[x][y] > Otsu_Threshold):
                    Img_Gray[x][y] = 255
                else:
                    Img_Gray[x][y]=0
        row_left, col_top, row_right, col_down=find_margin(Img_Gray,rows,columns)       #寻找外界矩形边界
        Img_Crop=Img_Gray[row_left:row_right+1,col_top:col_down+1]
        Resize_Img=Img_Create(Img_Crop,all_list[i])                #在Resize文档中创建该图像

        Img_TB=Drop_it_Top_Bottom(copy.deepcopy(Resize_Img))                     #从上向下滴水

        for x in range(Img_TB.shape[0]):
            for y in range(Img_TB.shape[1]):
                if Img_TB[x][y]==0:
                    print ('---' ,end=' ')
                else:
                    print(Img_TB[x][y],end=' ')
            print ('\n')

        cv2.imshow('fifi',Img_TB)
        cv2.waitKey(0)


        """def Scan_Line2(x0,y0,x1,y1,filename,Gray_img):       #(x0,y0)上滴点,(x1,y1)下滴点
    height, width = Gray_img.shape
    best_tpxy=[]                    #最优点
    candidate_tpxy=[]               #候选点
    tpxy=[]
    x2=x1-1;    y2=y1                   #寻找垂直点
    while(not Gray_img[x2][y2]==255):
        x2-=1
    L_line=max(abs(x2 - x0), abs(y2 - y0))
    dline_x = (x2 - x0) * 1.0 / L_line; dline_y = (y2 - y0) * 1.0 / L_line
    tpp_x=x0;   tpp_y=y0          #临时连线上的所有点依次遍历
    count_min = float("inf")  # 赋值无限大
    for i in range(0, L_line):
        tpp_x=tpp_x+dline_x; tpp_y=tpp_y+dline_y
        tp_x = round(tpp_x);   tp_y = round(tpp_y)            #临时点
        L=max(abs(tp_x-x1),abs(tp_y-y1))
        dx=(tp_x-x1)*1.0/L;   dy=(tp_y-y1)/L
        x=x1;   y=y1
        count=0;    check=0         #check用来检查是否w已经画出边界
        flag=1
        while(int(round(x))>=0 and int(round(x))<height and int(round(y))>=0 and int(round(y))<width):
            if(flag and Gray_img[int(round(x))][int(round(y))] == 0):       #Let`s Go!
                flag=0
            elif(flag and Gray_img[int(round(x))][int(round(y))] == 255):   #排除一开始的错切
                x = x + dx; y = y + dy
                continue

            elif (check == 1 and Gray_img[int(round(x))][int(round(y))] == 255):    #容忍度为1
                break
            elif(Gray_img[int(round(x))][int(round(y))] == 0):
                count+=1
                if(check>0):
                    check-=1
            elif(Gray_img[int(round(x))][int(round(y))] == 255):
                check+=1
            x = x + dx; y = y + dy
        if(count<=count_min):
            count_min=count
            try:
               tpxy.pop()
            except:
                pass
            tpxy.append((round(x), round(y),count))
        else:                                           #存入局部最小值
            if((tpxy[0][0],tpxy[0][1],tpxy[0][2]) not in candidate_tpxy):
                candidate_tpxy.append((tpxy[0][0],tpxy[0][1],tpxy[0][2]))
    print("*******************%r" %candidate_tpxy)

    count_min = float("inf")  # 赋值无限大
    for k in range(len(candidate_tpxy)):
        num=Plot_Line(candidate_tpxy[k][0], candidate_tpxy[k][1],x1,y1,filename, copy.deepcopy(Gray_img),0)
        if(num==1):
            continue
        elif(num==2 and candidate_tpxy[k][2]<count_min):
            count_min=candidate_tpxy[k][2]
            try:
                best_tpxy.pop()
            except:
                pass
            best_tpxy.append((candidate_tpxy[k][0], candidate_tpxy[k][1],candidate_tpxy[k][2]))
    if(len(best_tpxy)):
        Plot_Line(candidate_tpxy[k][0], candidate_tpxy[k][1],x1,y1, filename, copy.deepcopy(Gray_img),0)
        return best_tpxy[0][0],best_tpxy[0][1],x1,y1
    else:
        print("拒绝识别！")
        return 0,0,0,0
"""