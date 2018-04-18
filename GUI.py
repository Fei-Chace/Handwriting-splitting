# -*- encoding=UTF-8 -*-
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import f_Consecutive_dirp
import Clear
import Label as LB
import Normalization
import CNN

# from tkMessageBox import*

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("手写体数字识别")
        self.pack(fill=BOTH, expand=1)
        # 实例化一个Menu对象，这个在主窗体添加一个菜单
        menu = Menu(self.master)
        self.master.config(menu=menu)
        # 创建File菜单，下面有Save和Exit两个子菜单
        file = Menu(menu, tearoff=0)
        #file.add_command(label='打开图片', command=self.openPic)
        #file.add_command(label='打开文本', command=self.opentxt)
        file.add_command(label='Run', command=self.run)
        file.add_command(label='Label', command=self.Label)
        file.add_command(label='Normalization', command=self.Normalization)
        file.add_command(label='Exit', command=self.client_exit)
        menu.add_cascade(label='File', menu=file)

        Identify = Menu(menu, tearoff=0)
        Identify.add_command(label='CNN', command=self.CNN)
        menu.add_cascade(label='Ident', menu=Identify)

        Clear = Menu(menu, tearoff=0)
        Clear.add_command(label='Clear_1', command=self.Clear_1)
        Clear.add_command(label='Clear_2_4', command=self.Clear_2_4)
        Clear.add_command(label='Clear_5', command=self.Clear_5)
        Clear.add_command(label='Clear_6', command=self.Clear_6)
        menu.add_cascade(label='Clear', menu=Clear)
        # 创建Edit菜单，下面有一个Undo菜单
        #edit = Menu(menu, tearoff=0)
        #edit.add_command(label='Undo')
        #edit.add_command(label='Show  Image', command=self.showImg)
        #edit.add_command(label='Show  Text', command=self.showTxt)
        #menu.add_cascade(label='Edit', menu=edit)
        # 创建help菜单，有一个about按钮
        help = Menu(menu, tearoff=0)
        help.add_command(label="About", command=self.about)  # 用来传递参数
        menu.add_cascade(label='Edit', menu=help)

    def run(self):
        f_Consecutive_dirp.main()

    def Label(self):
        LB.main()

    def Normalization(self):
        Normalization.main()

    def Clear_2_4(self):
        Clear.Clear_2_4()

    def Clear_5(self):
        Clear.Clear_5()

    def Clear_1(self):
        Clear.Clear_1()

    def Clear_6(self):
        Clear.Clear_6()

    def CNN(self):
        correct, all, accuracy=CNN.main()
        w = Label(self, text="Accuracy为: %d/%d (%f%%)" %(correct, all, accuracy))
        w.pack(side=TOP)

    def about(self):
        w = Label(self, text="手写体数字识别\n费鼎淳\n10142130259\n")
        w.pack(side=TOP)

    def client_exit(self):
        exit()

root = Tk()
root.geometry("300x200")
app = Window(root)
root.mainloop()
"""
    def openPic(self):
        self.filename = askopenfilename()
        load = Image.open('%s' % self.filename)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0, anchor=NW)

    def opentxt(self):
        self.filename = askopenfilename(filetypes=[('TXT', 'txt')])
        label = []
        fp = open('%s' % self.filename)
        for line in fp:
            line.strip('\n')
            label.append(line.split(' '))
        label = label[0]  # 脱去一层[外壳]
        # 覆盖掉背景
        load = Image.open('E:/KNN/mnist/handwriting_pic/bg.jpg')
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0, anchor=NW)

        x = 0  # 横坐标放置位
        y = 0
        for i in range(len(label)):
            print (label[i])
            load = Image.open('C:/Users/conan/PycharmProjects/GUI_Testing/Num_Picture/' + str(label[i]) + '.png')
            print (load)
            render = ImageTk.PhotoImage(load)
            img = Label(self, image=render)
            img.image = render
            img.place(x=x, y=y, anchor=NW)
            x = x + 71
            if i % 10 == 0 and i != 0:
                y = y + 200
                x = 0

    def showImg(self):
        load = Image.open('E:/KNN/mnist/handwriting_pic/1.jpg')  # 我图片放桌面上
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0, anchor=NW)

    def showTxt(self):
        text = Label(self, text='GUI图形编程')
        text.pack()
"""

"""
# -*- encoding=UTF-8 -*-
from Tkinter import *
from PIL import Image, ImageTk

def hello():
    print('hello')

def about(master):
    w = Label(master,text="手写体数字识别\n费鼎淳\n10142130259\n")
    w.pack(side=TOP)

class App(Frame):
    def __init__(self, master):
        # 构造函数里传入一个父组件(master),创建一个Frame组件并显示
        menubar = Menu(master)
        # 创建下拉菜单File，然后将其加入到顶级的菜单栏中
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=hello)
        filemenu.add_command(label="Save", command=hello)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # 创建另一个下拉菜单Edit
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Show Image", command=self.showImg)
        editmenu.add_command(label="Copy", command=hello)
        editmenu.add_command(label="Paste", command=hello)
        menubar.add_cascade(label="Edit", menu=editmenu)
        # 创建下拉菜单Help
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=lambda : about(master))        #用来传递参数
        menubar.add_cascade(label="Help", menu=helpmenu)
        master.config(menu=menubar)
        # 创建两个button，并作为frame的一部分


        self.hi_there = Button(frame, text="Hello", command=self.say_hi)
        self.hi_there.pack(side=LEFT)
        self.button = Button(frame, text="QUIT", fg="red", command=frame.quit)
        self.button.pack(side=RIGHT)  # 此处side为LEFT表示将其放置 到frame剩余空间的最左方


    def say_hi(self):
        print "hi there, this is a class example!"

    def showImg(self):
        load = Image.open('E:/KNN/mnist/handwriting_pic/1.jpg')  # 我图片放桌面上
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

win = Tk()
win.title('数字识别程序')    #定义窗体标题
win.geometry('400x200')     #定义窗体的大小，是400X200像素
app = App(win)
win.mainloop()"""