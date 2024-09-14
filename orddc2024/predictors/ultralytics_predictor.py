from .base_predictor import Predictor
from ultralytics import YOLO

class UltralyticsPredictor(Predictor):
    def __init__(self, framework, models_params):
        super().__init__("ultralytics", framework)
        self.models_params = models_params

    def load_one_model(self, model_param):
        weight = model_param['weight']
        model = YOLO(weight)
        self.models.append((model, model_param))

    def predict_one_model(self, model, images, model_param):
        conf = model_param['conf']
        iou = model_param['iou']
        imgsz = model_param['img_size']
        augment = model_param['augment']
        agnostic_nms = model_param['agnostic_nms']
        
        results = model.predict(images, conf=conf, iou=iou, imgsz=imgsz, agnostic_nms=agnostic_nms)
        
        batch_boxes, batch_scores, batch_labels = [], [], []
        
        for result in results:
            boxes, scores, labels = [], [], []
            img_width, img_height = result.orig_shape[1], result.orig_shape[0]
            if result.boxes is not None:
                for box in result.boxes.data.tolist():
                    cls = int(box[5])
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    score = box[4]
                    normalized_box = self.normalize_box([x1, y1, x2, y2], img_width, img_height)
                    boxes.append(normalized_box)
                    scores.append(score)
                    labels.append(cls + 1)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_labels.append(labels)
        # print("batch_boxes:", batch_boxes)
        # print("batch_scores:", batch_scores)
        # print("batch_labels:", batch_labels)
        return batch_boxes, batch_scores, batch_labels

    def predict(self, batch_size=128):
        boxes_list = []
        scores_list = []
        labels_list = []
        
        num_batches = len(self.images) // batch_size + (1 if len(self.images) % batch_size != 0 else 0)

        for i in range(num_batches):
            batch_images = self.images[i*batch_size : (i+1)*batch_size]
            # batch_boxes, batch_scores, batch_labels = [], [], []
            for model, model_param in self.models:
                model_boxes, model_scores, model_labels = self.predict_one_model(model, batch_images, model_param)
                boxes_list.extend(model_boxes)
                scores_list.extend(model_scores)
                labels_list.extend(model_labels)
            # boxes_list.extend(batch_boxes)
            # scores_list.extend(batch_scores)
            # labels_list.extend(batch_labels)
        # print('boxes_list: ', boxes_list)
        # print('scores_list: ', scores_list)        
        # print('labels_list: ', labels_list)        
                
        return [boxes_list], [scores_list], [labels_list]
    
    def load(self, models_params, images_path):
        self.images = self.load_images(images_path)
        for model_param in models_params:
            print(f"Loading model from weight: {model_param['weight']}")
            self.load_one_model(model_param)
            # print(f"Current models: {self.models}")