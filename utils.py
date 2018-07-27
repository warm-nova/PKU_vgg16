#coding:utf-8

from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

#正常显示中文和符号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def load_image(path):
    fig = plt.figure("Center and Resize")
    img = io.imread(path)
    img = img / 255.0

    ax0 = fig.add_subplot(131)
    ax0.set_xlabel(u'Original Picture')
    ax0.imshow(img)
    #展示这张图片

    short_edge = min(img.shape[:2])
    y = int((img.shape[0]-short_edge) / 2)
    x = int((img.shape[1]-short_edge) / 2)
    crop_img = img[y:y+short_edge,x:x+short_edge]
    print(crop_img.shape)

    ax1 = fig.add_subplot(132)
    ax1.set_xlabel(u'Center Picture')
    ax1.imshow(crop_img)
    #转换成224*224
    re_img = transform.resize(crop_img,(224,224))

    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u'Resize Picture')
    ax2.imshow(re_img)
    img_ready = re_img.reshape((1,224,224,3))
    return img_ready

def percent(value):
    return '%.2f%%' % (value*100)


