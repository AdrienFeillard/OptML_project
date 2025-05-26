from enum import Enum

class Classifier(str, Enum):
    vgg11_bn = "vgg11_bn"
    vgg13_bn = "vgg13_bn"
    vgg16_bn = "vgg16_bn"
    vgg19_bn = "vgg19_bn"
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    densenet121 = "densenet121"
    densenet161 = "densenet161"
    densenet169 = "densenet169"
    mobilenet_v2 = "mobilenet_v2"
    googlenet = "googlenet"
    inception_v3 = "inception_v3"
    simple_cnn = "simple_cnn"
    tiny_cnn = "tiny_cnn"

class LoggerType(str, Enum):
    tensorboard = "tensorboard"
    wandb = "wandb"

class NoiseDistribution(str, Enum):
    gaussian = "gaussian"
    uniform = "uniform"

# Theme for visualization
THEME = {
    "title": "magenta",
    "heading": "yellow",
    "metrics": "cyan",
    "good": "green",
    "warning": "yellow",
    "bad": "red",
    "accent": "blue",
    "graph_dots": "bright_cyan",
    "graph_line": "blue"
}

# Architecture information for display
ARCHITECTURE_INFO = {
    "vgg11_bn": {"type": "VGG", "layers": 11, "params": "132.9M", "description": "VGG-11 with batch normalization"},
    "vgg13_bn": {"type": "VGG", "layers": 13, "params": "133.1M", "description": "VGG-13 with batch normalization"},
    "vgg16_bn": {"type": "VGG", "layers": 16, "params": "138.4M", "description": "VGG-16 with batch normalization"},
    "vgg19_bn": {"type": "VGG", "layers": 19, "params": "143.7M", "description": "VGG-19 with batch normalization"},
    "resnet18": {"type": "ResNet", "layers": 18, "params": "11.7M", "description": "ResNet with 18 layers"},
    "resnet34": {"type": "ResNet", "layers": 34, "params": "21.8M", "description": "ResNet with 34 layers"},
    "resnet50": {"type": "ResNet", "layers": 50, "params": "25.6M", "description": "ResNet with 50 layers"},
    "densenet121": {"type": "DenseNet", "layers": 121, "params": "8.0M", "description": "DenseNet with 121 layers"},
    "densenet161": {"type": "DenseNet", "layers": 161, "params": "28.7M", "description": "DenseNet with 161 layers"},
    "densenet169": {"type": "DenseNet", "layers": 169, "params": "14.2M", "description": "DenseNet with 169 layers"},
    "mobilenet_v2": {"type": "MobileNet", "layers": "-", "params": "3.5M", "description": "MobileNetV2"},
    "googlenet": {"type": "GoogleNet", "layers": "-", "params": "6.8M", "description": "GoogLeNet/Inception v1"},
    "inception_v3": {"type": "Inception", "layers": "-", "params": "27.2M", "description": "Inception v3"},
}