# coding:utf-8
from __future__ import division
import colorsys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import ImageDraw, ImageFont
from nets.centernet import CenterNet_Resnet50
from utils.utils import decode_bbox, centernet_correct_boxes, letterbox_image, nms

class CenterNet(object):
    _defaults = {
        "model_path"    : 'model_data/centernet_resnet50_voc.pth',
        "classes_path"  : 'model_data/classesvoc.txt',
        "image_size"    : [512, 512, 3],
        "confidence"    : 0.3,
        # backbone为resnet50时建议nms为True, hourglass时建议为False
        "nms"           : True,
        "nms_threhold"  : 0.3,
        "cuda"          : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化centernet
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # 获得所有分类
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        self.num_classes = len(self.class_names)
        self.centernet = CenterNet_Resnet50(num_classes=self.num_classes, pretrain=False)

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.centernet.load_state_dict(state_dict, strict=True)
        self.centernet = self.centernet.eval()


        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.centernet = nn.DataParallel(self.centernet)
            self.centernet.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # 检测图片
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = letterbox_image(image, [self.image_size[0], self.image_size[1]])
        # 网上训练好的这个模型用的是BGR而不是RGB
        photo = np.array(crop_img, dtype=np.float32)[:, :, ::-1]
        # photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.
        photo = np.transpose(photo, (2, 0, 1))
        images = []
        images.append(photo)

        images = np.asarray(images)
        images = torch.from_numpy(images)

        if self.cuda:
            images = images.cuda()

        with torch.no_grad():
            outputs = self.centernet(images)
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

        try:
            if self.nms:
                outputs = np.array(nms(outputs, self.nms_threhold))
        except:
            pass

        output = outputs[0]
        if len(output) <= 0:
            return image

        batch_boxes, det_conf, det_label = output[:, :4], output[:, 4], output[:, 5]
        det_xmin, det_ymin, det_xmax, det_ymax = batch_boxes[:, 0], batch_boxes[:, 1], batch_boxes[:, 2], batch_boxes[:, 3]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = np.expand_dims(det_xmin[top_indices], -1)
        top_ymin = np.expand_dims(det_ymin[top_indices], -1)
        top_xmax = np.expand_dims(det_xmax[top_indices], -1)
        top_ymax = np.expand_dims(det_ymax[top_indices], -1)

        # 去掉灰条
        boxes = centernet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                        np.array([self.image_size[0], self.image_size[1]]),
                                        image_shape)
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(1e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        # thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            # top = top - 5
            # left = left - 5
            # bottom = bottom + 5
            # right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # for i in range(thickness):
            for i in range(2):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label), fill=(0, 0, 0), font=font)
            del draw

        return image