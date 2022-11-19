import argparse
import os
from re import I
import sys
from pathlib import Path
from tkinter import W
from cv2 import IMWRITE_PAM_FORMAT_RGB_ALPHA
import matplotlib.pyplot as plt
from yaml import parse
from extract_images import extractImages
sys.path.append(os.path.join(os.path.dirname(__file__), '../../')) # root

import onnxruntime as ort
from tqdm import tqdm
import numpy as np
import cv2

from glob import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

class yolov5onnx():
    def __init__(self, model_path, nc = 13, model_hw=[640,640], threshold=0.4, iou=0.5, nms_threshold=0.7, batch_size=1, mem_limit=800):
        """
        YOLOv5-tiny onnx object.\n
        Send imagecv2 to `pre_processing`, call `inference` to net forward and get yolo type detections from `post_processing`
        Args:\n
            model_path: path to onnx model\n
            classes: list of classes\n
            model_hw: model input height and width\n
            threshold: confidence threshold for object accuracy\n
            iou: for non-max suppression\n
            nms_threshold: for non-max suppression\n
            batch_size: batch size\n
            mem_limit: memory limit for onnx runtime\n
        """
        self.model_path = model_path
        self.model_hw = model_hw
        self.model_input_img = np.zeros((1, 3, self.model_hw[0], self.model_hw[1]), dtype=np.float32)
        self.threshold = threshold
        self.iou = iou
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
        self.nc = nc
        providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': mem_limit * 720 * 1280,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
        ]
        self.model = ort.InferenceSession(self.model_path,providers=providers)
        self.input_name = self.model.get_inputs()[0].name

        

    def _letterbox_convert(self):
        """
        Adjust the size of the frame from the webcam to the model input shape.
        """
        f_height, f_width = self.imgcv2.shape[0], self.imgcv2.shape[1]
        scale = np.max((f_height / self.model_hw[0], f_width / self.model_hw[1]))

        # padding base
        img = np.zeros(
            (int(round(scale * self.model_hw[0])), int(round(scale * self.model_hw[1])), 3),
            np.uint8
        )
        start = (np.array(img.shape) - np.array(self.imgcv2.shape)) // 2
        img[
            start[0]: start[0] + f_height,
            start[1]: start[1] + f_width
        ] = self.imgcv2
        self.model_input_img = cv2.resize(img, (self.model_hw[1], self.model_hw[0]))

    def _reverse_letterbox(self, detections):
        h, w = self.imgcv2.shape[0], self.imgcv2.shape[1]

        pad_x = pad_y = 0
        if self.model_hw != None:
            scale = np.max((h / self.model_hw[0], w / self.model_hw[1]))
            start = (self.model_hw[0:2] - np.array(self.imgcv2.shape[0:2]) / scale) // 2
            pad_x = start[1]*scale
            pad_y = start[0]*scale

        self.yolos = []
        for detection in detections:
            obj = [detection[0],
                # detection[1],
                (detection[0]*(w+pad_x*2) - pad_x)/w,
                (detection[1]*(h+pad_y*2) - pad_y)/h,
                (detection[2]*(w+pad_x*2))/w,
                (detection[3]*(h+pad_y*2))/h,
            ]
            obj[1] = obj[1]+obj[3]/2
            obj[2] = obj[2]+obj[4]/2
            self.yolos.append(obj)
        return self.yolos

    def _bbox_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area1 = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None)
        inter_area2 = np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

        inter_area = inter_area1 * inter_area2

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def _nms(self, prediction):
        box_corner = np.zeros(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2 # cx - w/2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2 # cy - h/2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2 # cx + w/2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2 # cy + h/2
        prediction[:, :, :4] = box_corner[:, :, :4] #conf

        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            conf_mask = (image_pred[:, 4] >= self.threshold).squeeze()
            image_pred = image_pred[conf_mask]

            if not image_pred.shape[0]:
                continue

            class_conf = np.max(image_pred[:, 5:5 + self.nc], axis=1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 5:5 + self.nc], axis=1)
            class_pred = class_pred.reshape((class_pred.shape[0],1))
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)
            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]

                conf_sort_value = np.sort(detections_class[:, 4])
                conf_sort_value = conf_sort_value[::-1]

                conf_sort_index = np.argsort(detections_class[:, 4])
                conf_sort_index = conf_sort_index[::-1]

                detections_class = detections_class[conf_sort_index]

                max_detections = []
                while detections_class.shape[0]:
                    expand_detections_class = np.expand_dims(detections_class[0],0)
                    max_detections.append(expand_detections_class)
                    if len(detections_class) == 1:
                        break
                    ious = self._bbox_iou(expand_detections_class, detections_class[1:])
                    detections_class = detections_class[1:][ious < self.nms_threshold]

                max_detections = np.concatenate(max_detections)
                
                output[image_i] = max_detections if output[image_i] is None else np.concatenate(
                    (output[image_i], max_detections))

        return output

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _hsv_to_rgb(self, h, s, v):
        bgr = cv2.cvtColor(
            np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)

    def _imread(self, filename, flags=cv2.IMREAD_COLOR):
        if not os.path.isfile(filename):
            sys.exit()
        data = np.fromfile(filename, np.int8)
        img = cv2.imdecode(data, flags)
        return img

    def pre_processing(self, imgcv2):
        # if os.path.isfile(img_path):
        #     self.img = self._imread(img_path, cv2.IMREAD_UNCHANGED)
        # else:
        #     # print "Image path is not valid" as a red text
        #     print("\033[1;31mimage path is not valid. skipping...\033[0m")
        self.imgcv2 = imgcv2
        if len(self.imgcv2.shape) < 3 or self.imgcv2.shape[2] == 1:
            self.imgcv2 = cv2.cvtColor(self.imgcv2, cv2.COLOR_GRAY2BGR)
        elif self.imgcv2.shape[2] == 4:
            self.imgcv2 = cv2.cvtColor(self.imgcv2, cv2.COLOR_BGRA2BGR)
        
        self._letterbox_convert()
        self.model_input = cv2.cvtColor(self.model_input_img, cv2.COLOR_BGR2RGB)
        self.model_input = np.transpose(self.model_input, [2, 0, 1])
        self.model_input = self.model_input.astype(np.float32) / 255
        self.model_input = np.expand_dims(self.model_input, 0)
        
    def inference(self):
        self.outputs = self.model.run(None, {self.input_name: self.model_input})

        
    def post_processing(self, img):
        batch_detections = []
        count = 0
        img_hw = img.shape[:2]

        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

        boxs = []
        a = np.array(anchors).reshape(3, -1, 2)
        anchor_grid = a.copy().reshape(3, 1, -1, 1, 1, 2)

        if len(self.outputs)==1:
            # yolov5 v6
            outputx = self.outputs[0]
        else:
            # yolov5 v1

            #(1, 3, 80, 80, 85) # anchor 0
            #(1, 3, 40, 40, 85) # anchor 1
            #(1, 3, 20, 20, 85) # anchor 2

            #[cx,cy,w,h,conf,pred_cls(80)]

            for index, out in enumerate(self.outputs):
                batch = out.shape[1]
                feature_h = out.shape[2]
                feature_w = out.shape[3]

                # Feature map corresponds to the original image zoom factor
                stride_w = int(self.model_hw[1] / feature_w)
                stride_h = int(self.model_hw[0] / feature_h)

                grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

                # cx, cy, w, h
                pred_boxes = np.zeros(out[..., :4].shape)
                pred_boxes[..., 0] = (self._sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
                pred_boxes[..., 1] = (self._sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
                pred_boxes[..., 2:4] = (self._sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[index]  # wh
                

                conf = self._sigmoid(out[..., 4])
                pred_cls = self._sigmoid(out[..., 5:])

                output = np.concatenate((pred_boxes.reshape(self.batch_size, -1, 4),
                                    conf.reshape(self.batch_size, -1, 1),
                                    pred_cls.reshape(self.batch_size, -1, self.nc)),
                                    -1)
                boxs.append(output)

            outputx = np.concatenate(boxs, 1)
        # NMS
        batch_detections = self._nms(outputx)

        detections = batch_detections[0]
        if detections is None:
            return [[]]

        labels = detections[..., -1]
        boxs = detections[..., :4]
        confs = detections[..., 4]

        bboxes = []

        bboxes_batch = []
        
        for i, box in enumerate(boxs):
            x1, y1, x2, y2 = box
            c = int(labels[i])
            r = [
                x1/self.model_hw[1],
                y1/self.model_hw[0],
                (x2 - x1)/self.model_hw[1],
                (y2 - y1)/self.model_hw[0],
            ]
          
            bboxes.append(r)
            bboxes_batch.append(bboxes)
            org_bboxes = self._reverse_letterbox(bboxes_batch[0])

            xa = int(org_bboxes[i][1]*img_hw[1]) - int(org_bboxes[i][3]*img_hw[1] / 2)
            ya = int(org_bboxes[i][2]*img_hw[0]) - int(org_bboxes[i][4]*img_hw[0] / 2)
            xa2 = int(org_bboxes[i][1]*img_hw[1]) + int(org_bboxes[0][3]*img_hw[1] / 2)
            ya2 = int(org_bboxes[i][2]*img_hw[0]) + int(org_bboxes[i][4]*img_hw[0] / 2)

            if xa > 0:
                xa = int(org_bboxes[i][1]*img_hw[1]) - int(org_bboxes[i][3]*img_hw[1] / 2)
            elif xa > img_hw[1]:
                xa = img_hw[1]
            else:
                xa = 0

            if ya > 0:
                ya = int(org_bboxes[i][2]*img_hw[0]) - int(org_bboxes[i][4]*img_hw[0] / 2)
            elif ya > img_hw[0]:
                ya = img_hw[0]
            else:
                ya = 0

            if xa2 > 0:
                xa2 = int(org_bboxes[i][1]*img_hw[1]) + int(org_bboxes[0][3]*img_hw[1] / 2)
            elif xa2 > img_hw[1]:
                xa2 = img_hw[1]
            else:
                xa2 = 0

            if ya2 > 0:
                ya2 = int(org_bboxes[i][2]*img_hw[0]) + int(org_bboxes[i][4]*img_hw[0] / 2)
            elif ya2 > img_hw[0]:
                ya2 = img_hw[0]
            else:
                ya2 = 0

            
            # cv2.rectangle(img, (xa2,ya2), (xa,ya), color = (255,0,0), thickness=(4))
          
            roi = img[ya:ya2, xa:xa2]
            roi = cv2.blur(roi, (23,23), 0)
            img[ya:ya+roi.shape[0], xa:xa+roi.shape[1]] = roi
        

 

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--modelPath", default="./weights/best.onnx", help="path to model")
    a.add_argument("--videoPath", default="./datasets/test_videos/test5.mp4", help="path to input video")
    a.add_argument("--outputVideo", default="./datasets/test_results/", help="path to output")
    args = a.parse_args()

    count = 0
    img_array = []
    
    model_file = args.modelPath
    img_size = [640,640]
    vidcap = cv2.VideoCapture(args.videoPath)
    success,image = vidcap.read()
    success = True
    if (vidcap.isOpened() == False):
        print("Error opening the video file")
    else:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("Frames per second: ", fps, 'FPS')

    while tqdm(success):
        success,image = vidcap.read()
        # print ('Read a new frame: ', success)
        height, width, layers = image.shape
        size = (width,height)    
        img_array.append(image)

        om = yolov5onnx(model_file, nc=1, model_hw=img_size, mem_limit=800)
        om.pre_processing(image)
        om.inference()
        image = om.post_processing(image)
        
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(args.outputVideo + 'project.avi',fourcc, 29.97002997002997, size)
            
        for i in range(0,len(img_array)):
            out.write(img_array[i])
        out.release()
  




