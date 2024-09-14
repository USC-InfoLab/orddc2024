from abc import ABC, abstractmethod
import os
from PIL import Image

class Predictor(ABC):
    def __init__(self, repo, framework):
        self.repo = repo
        self.framework = framework
        self.models = []

    def load(self, weights, images_path):
        self.weights = weights
        self.images = self.load_images(images_path)
        for weight in weights:
            print(f"Loading model from weight: {weight}")
            self.load_one_model(weight)
            print(f"Current models: {self.models}")

    def load_images(self, images_path):
        if os.path.isdir(images_path):
            return [os.path.join(images_path, img) for img in os.listdir(images_path) if img.endswith('.jpg') or img.endswith('.png')]
        elif os.path.isfile(images_path):
            with open(images_path, 'r') as file:
                return [line.strip() for line in file.readlines()]
        else:
            raise ValueError("Invalid path to images. Provide a directory or a text file with image paths.")

    @abstractmethod
    def load_one_model(self, weight):
        pass

    @abstractmethod
    def predict_one_model(self, model, image):
        pass

    @staticmethod
    def normalize_box(box, img_width, img_height):
        x1, y1, x2, y2 = box
        x1 /= img_width
        y1 /= img_height
        x2 /= img_width
        y2 /= img_height
        return [x1, y1, x2, y2]

    @staticmethod
    def denormalize_box(box, img_width, img_height):
        x1, y1, x2, y2 = box
        x1 *= img_width
        y1 *= img_height
        x2 *= img_width
        y2 *= img_height
        return [x1, y1, x2, y2]
    
    def get_image_size(self, image_path):
        with Image.open(image_path) as img:
            return img.size

    def predict(self):
        boxes_list = []
        scores_list = []
        labels_list = []
        for model in self.models:
            model_boxes = []
            model_scores = []
            model_labels = []
            for image in self.images:
                b, s, l = self.predict_one_model(model, image)
                model_boxes.append(b)
                model_scores.append(s)
                model_labels.append(l)
            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)
        return boxes_list, scores_list, labels_list
