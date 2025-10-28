import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


"""
Define model class here

- Want to detect semantic entities in images, so an object detection model would be appropriate, such as ResNet with Faster R-CNN.
- Multiple objects per image, bounding boxes around each object, and classification of each object, Feature Pyramid Network (FPN) would be useful.
- 2000 samples, not too huge of a dataset.

https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn

"""


class DetectionModel(torch.nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.classes = classes
        # use pretrained model and edit classifier head to fit detection goals (transfer learning)
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        except ImportError:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True) # fallback for older torchvision versions, just in case
        # changing the classifier head to match number of classes (3 + background)
        input_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(input_features, classes)
        

    def forward(self, imgs, targets=None):
        if self.training:
            return self.model(imgs, targets)
        else: # eval mode
            return self.model(imgs)
        

    def predict(self, imgs, conf_threshold=0.5):
        self.eval()
        with torch.no_grad():
            predictions = self.model(imgs)
        
        # Apply confidence thresholding
        filtered_preds = []
        for pred in predictions:
            keep = pred['scores'] >= conf_threshold   # if score is above threshold, keep the box
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_preds.append(filtered_pred)
        return filtered_preds