# coding: utf-8
import cv2
import time
import numpy as np
from PIL import Image
from centernet import CenterNet

centernet = CenterNet()

capture = cv2.VideoCapture(0)

fps = 0.0

while True:
    t1 = time.time()
    # 读取一帧
    ref, frame = capture.read()
    # 格式转变 BGR2RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(centernet.detect_image(frame))

    # RGB2BGR 满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    fps = (fps + (1./(time.time() - t1))) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('mp4', frame)

    c = cv2.waitKey(30) & 0xff

    if c == 27:
        capture.release()
        break