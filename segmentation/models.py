from torch import nn
from torch.nn.functional import relu, sigmoid, tanh
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
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
        self.model = deeplabv3_resnet50(weights=None, num_classes=2)

    def forward(self, x):
        return self.model(x)["out"]

    def name(self):
        return "deeplabv3_resnet50_crack"


class CrackSegmentationModelMN32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)

    def forward(self, x):
        return self.model(x)["out"]

    def name(self):
        return "deeplabv3_mobilenet_v3_large_crack"


class CrackSegmentationModelCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def name(self):
        return "custom"


CrackSegmentationModel = CrackSegmentationModelMN32
