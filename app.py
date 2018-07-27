#coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#diy
import vgg16
import utils
from Nclasses import labels

img_path = input("Input the path and image name:")
img_ready = utils.load_image(img_path)

fig=plt.figure(u"Top 5 预测结果")

with tf.Session() as sess:
    x = tf.placeholder(tf.float32,[1,224,224,3])
    vgg = vgg16.Vgg16()

    vgg.forward(x)

    probablity = sess.run(vgg.prob, feed_dict={x:img_ready})

    top5= np.argsort(probablity[0])[-1:-6:-1]
    print("top5",top5)

values = []
bar_label = []

for n,i in enumerate(top5):
    print("n:",n)
    print("i:",i)
    values.append(probablity[0][i])
    bar_label.append(labels[i])
    print(i,":",labels[i],"-----",utils.percent(probablity[0][i]))

ax = fig.add_subplot(111)
ax.bar(range(len(values)),values,tick_label=bar_label,linewidth=0.5)
ax.set_ylabel(u"probability")
ax.set_title("TOP5")
for a,b in zip(range(len(values)),values):
    ax.text(a,b+0.0005,utils.percent(b),ha="center",va='bottom',fontsize=7)

plt.savefig('./result.jpg')
plt.show()