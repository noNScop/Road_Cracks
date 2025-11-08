from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class CrackSegmentationModelDL101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
        )
        self.model.classifier = DeepLabHead(2048, 2)  # 2 classes: background, crack

    def forward(self, x):
        return self.model(x)["out"]

    def name(self):
        return "deeplabv3_resnet101_crack"


class CrackSegmentationModelDL50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        )
        self.model.classifier = DeepLabHead(2048, 2)  # 2 classes: background, crack

    def forward(self, x):
        return self.model(x)["out"]

    def name(self):
        return "deeplabv3_resnet50_crack"


CrackSegmentationModel = CrackSegmentationModelDL50
