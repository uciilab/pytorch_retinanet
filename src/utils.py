from resources.config import COCO_CLASS_NAMES as coco_names
from resources.config import retinanet_min_size
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch
import cv2


class Object_detector:
    def __init__(self):
        self.model = None
        self.colors = None
        self.transform = None
        self.device = None

    def set_colors(self, class_names):
        self.colors = np.random.uniform(0, 255, size=(len(coco_names), 3))

    def set_transform(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def set_model(self, model_name):
        if model_name == 'retinanet':
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True,
                                                                             min_size=retinanet_min_size)

    def set_device(self):
        pass

    # Create predict function
    def predict(self, image, threshold):
        pass

    # Create draw BBox function
    def create_bounding_box(self, image, classes):
        pass
