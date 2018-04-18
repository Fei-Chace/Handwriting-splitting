import os

def Clear_2_4():
    all_list = os.listdir('./img_2_Resize/')
    for i in range(len(all_list)):
        filename=('./img_2_Resize/%s' %all_list[i])
        os.remove(filename)

    all_list = os.listdir('./img_3_Plot/')
    for i in range(len(all_list)):
        filename=('./img_3_Plot/%s' %all_list[i])
        os.remove(filename)

    all_list = os.listdir('./img_4_Split/')
    for i in range(len(all_list)):
        filename=('./img_4_Split/%s' %all_list[i])
        os.remove(filename)


def Clear_5():
    all_list = os.listdir('./img_5_Label/')
    for i in range(len(all_list)):
        filename=('./img_5_Label/%s' %all_list[i])
        os.remove(filename)

def Clear_1():
    all_list = os.listdir('./img_1_Raw/')
    for i in range(len(all_list)):
        filename = ('./img_1_Raw/%s' % all_list[i])
        os.remove(filename)

def Clear_6():
    all_list = os.listdir('./img_6_Normalization/')
    for i in range(len(all_list)):
        file_list=os.listdir('./img_6_Normalization/%s' %all_list[i])
        for j in range(len(file_list)):
            filename=('./img_6_Normalization/%s/%s' %(all_list[i],file_list[j]))
            os.remove(filename)

if __name__ == '__main__':
    Clear_2_5()
    #Clear_1()
    #Clear_6()

