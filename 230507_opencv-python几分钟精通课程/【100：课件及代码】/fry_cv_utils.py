
"""
版本更新：
    v12: 截取函数变化
    v13: 修正【截取中心区域】的算法的bug
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import copy
import os
from IPython.core.pylabtools import figsize # import figsize

from PIL import Image, ImageDraw, ImageFont




class FryCvUtils():


    @staticmethod
    def plt_show_img(img_list,row_num=4,fig_size=(2,2),title="",saveImgPath="",imgType="BGR"):
        """
        作用：
            用matplotlib直接打印四张图片
        Args:
            img_list：被imshow的图片内容，是一个列表
            row_num：每行显示的图片数量，默认是4
            fig_size：
                    # 设置 figsize，
                    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
                    # 指定dpi=200，图片尺寸为 1200*800
                    # 指定dpi=300，图片尺寸为 1800*1200
                    # 设置figsize可以在不改变分辨率情况下改变比例
        Returns:
            None
        """

        # 设置matplotlib库字体的非衬线字体为黑体
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        # 设置matplotlib库字体族为非衬线字体
        plt.rcParams["font.family"] = "sans-serif"

        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率

        textFont = {
            'color': 'blueviolet',
            'size': fig_size[0] * 3,
        }

        # plt.title(title, fontsize=fig_size[0]*3)
        # plt.text(0, 0, r' ' + title, fontdict={'size': fig_size[0]*3, 'color': 'blueviolet'})

        figsize(fig_size[0], fig_size[1])  # 设置 figsize

        # 第一步：把图片从BGR转换成RGB模式
        new_img_list = []
        imgCount = 0
        for imgI in img_list:
            assert (type(imgI) is np.ndarray), "图片只能是numpy类型"
            assert (imgI.ndim > 1), "图片必须是2维及以上"
            ansImgI = imgI
            # print("imgI.ndim :{}".format(imgI.ndim))
            if imgI.ndim == 1:
                ansImgI = imgI
            if imgI.ndim == 2:
                ansImgI = imgI
            if imgI.ndim == 3:
                # img_rgb2 = i[:,:,::-1]
                # 230324：opencv 报错‘depth‘ is 6 (CV_64F)全因numpy 默认float类型是float64位
                # 转成32的float就好
                ansImgI = imgI
                if ansImgI.dtype == "float64":
                    ansImgI = ansImgI.astype("float32")
                # 230425：
                # 如果图片是BGR，默认转换了一道，所以我读BGR的图片是正常显示，而读RGB的图片就是异常显示了
                if "BGR" == imgType:
                    ansImgI = cv2.cvtColor(ansImgI, cv2.COLOR_BGR2RGB)
            if imgI.ndim == 4:
                print("第 {} 张图片是4个通道".format(imgCount))
            new_img_list.append(ansImgI)
            imgCount += 1

        img_list = new_img_list

        num = len(img_list)  # 总的显示图片的数量
        # 总行数
        rows = math.ceil(num/row_num)
        # 每行的元素个数
        per_row_num = []
        num1 = num
        while num1 > row_num:
            num1 = num1-row_num
            per_row_num.append(row_num)
        if(num1 > 0):
            per_row_num.append(num1)
        # print(per_row_num)

        fig, ax = plt.subplots()

        len1 = len(per_row_num)
        k = 0
        for i in range(1, len1+1):
            row_num2 = per_row_num[i-1]
            for j in range(1, row_num2+1):
                # print(len1,row_num,k+1)
                ax = plt.subplot(len1, row_num, k+1)
                
                # 如果有title才写title
                if len(title):
                    textStr = r' ' + title+"_"+str(k)
                    plt.text(-0.1, -0.4, r' ' + title+"_"+str(k), fontdict=textFont)
                
                # 如果是uint8,并且是3个通道
                if img_list[k].dtype == 'uint8' and img_list[k].ndim == 3:
                    plt.imshow(img_list[k])
                else:
                    plt.imshow(img_list[k], cmap='gray')
                # 如果是double类型或者是浮点类型
                k += 1
        
        # 230328：保存图片
        if len(saveImgPath):
            plt.savefig(saveImgPath)
        
        plt.show()


if __name__ == "__main__":
    
    img = np.random.randint(100,255,(320,320,3), np.uint8)
    img2 = np.random.randint(0,255,(320,320,3), np.uint8)
    # print(img.shape) # 输出：(320, 320, 3)

    
    # ! 测试plt显示图片
    img_list=[img,img2,img]
    FryCvUtils.plt_show_img(img_list,2)
    





