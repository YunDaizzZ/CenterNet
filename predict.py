# coding:utf-8
# 对单张图片进行预测

from centernet import CenterNet
from PIL import Image

centernet = CenterNet()

img = '/home/bhap/Pytorch_test/CenterNet/img/1.jpg'
image = Image.open(img)
r_image = centernet.detect_image(image)
r_image.show()