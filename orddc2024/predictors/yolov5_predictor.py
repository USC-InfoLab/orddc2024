from .base_predictor import Predictor
import torch
import os
import cv2
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5')))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

class Yolov5Predictor(Predictor):
    def __init__(self, framework, models_params):
        super().__init__("yolov5", framework)
        self.models_params = models_params
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

    def load_one_model(self, model_param):
        weight = model_param['weight']
        model = DetectMultiBackend(weight, device=self.device)
        model_param['img_size'] = check_img_size(model_param['img_size'], s=model.stride)  # check img_size
        if self.half:
            model.model.half()  # to FP16
        self.models.append((model, model_param))

    def predict_one_model(self, model, image_path, model_param):
        dataset = LoadImages(image_path, img_size=model_param['img_size'], stride=model.stride, auto=model.pt)
        boxes = []
        scores = []
        labels = []
        
        for path, img, im0s, vid_cap, s in dataset:
            print(f"Processing image: {path}")
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=model_param['augment'], visualize=False)
            pred = non_max_suppression(pred, model_param['conf'], model_param['iou'], classes=None, agnostic=model_param['agnostic_nms'])

            for det in pred:  # detections per image
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = [val.detach().cpu().item() for val in xyxy]
                        x1, y1, x2, y2 = xyxy
                        score = conf
                        cls_id = cls
                        boxes.append(self.normalize_box([x1, y1, x2, y2], im0s.shape[1], im0s.shape[0]))
                        scores.append(score.detach().cpu().item())
                        labels.append(cls_id.detach().cpu().item() + 1)
        
        return boxes, scores, labels

    def predict(self):
        boxes_list = []
        scores_list = []
        labels_list = []
        for model, model_param in self.models:
            model_boxes = []
            model_scores = []
            model_labels = []
            for image in self.images:
                b, s, l = self.predict_one_model(model, image, model_param)
                model_boxes.append(b)
                model_scores.append(s)
                model_labels.append(l)
            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)
        return boxes_list, scores_list, labels_list

    def load(self, models_params, images_path):
        self.images = self.load_images(images_path)
        for model_param in models_params:
            print(f"Loading model from weight: {model_param['weight']}")
            self.load_one_model(model_param)
            # print(f"Current models: {self.models}")