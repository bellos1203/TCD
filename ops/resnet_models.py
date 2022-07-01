import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def gradhook(self, grad_input, grad_output):
    importance = grad_output[0] ** 2 # [N, (T), C, H, W]
    if len(importance.shape) >= 4:
        importance = torch.sum(importance, -1) # [N, (T), C, H]
        importance = torch.sum(importance, -1) # [N, (T), C]
    if self.num_segments == 8:
        importance = importance.view(-1,self.num_segments,importance.size(-1)) # [N, T, C]
    importance = torch.mean(importance, 0) # [C]
    self.importance += importance


class Channel_Importance_Measure(nn.Module):
    def __init__(self, num_channels, num_segments=1):
        super().__init__()
        self.num_channels = num_channels
        self.num_segments = num_segments
        self.scale = nn.Parameter(torch.randn(num_channels), requires_grad=False)
        nn.init.constant_(self.scale, 1.0)
        if self.num_segments==1:
            self.register_buffer('importance', torch.zeros_like(self.scale))
        else:
            self.register_buffer('importance', torch.zeros(self.num_segments, self.num_channels))

    def forward(self, x):
        if len(x.shape) == 4:
            x = x * self.scale.reshape([1,-1,1,1])
        else:
            x = x * self.scale.reshape([1,-1])
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_relu = remove_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if not self.remove_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remove_relu = remove_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if not self.remove_relu:
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, remove_last_relu, num_segments=8, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, remove_relu=self.remove_last_relu)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.layer1_importance = Channel_Importance_Measure(64 * block.expansion, num_segments)
        self.layer2_importance = Channel_Importance_Measure(128 * block.expansion, num_segments)
        self.layer3_importance = Channel_Importance_Measure(256 * block.expansion, num_segments)
        self.layer4_importance = Channel_Importance_Measure(512 * block.expansion, num_segments)
        self.raw_features_importance = Channel_Importance_Measure(512 * block.expansion, num_segments)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, remove_relu=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        if remove_relu:
            layers.append(block(self.inplanes, planes, remove_relu=remove_relu))
        else:
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        int_features = []
        x = self.layer1(x)
        int_features.append(x)
        _,C,H,W = x.shape
        x = self.layer1_importance(x)

        x = self.layer2(x)
        int_features.append(x)
        _,C,H,W = x.shape
        x = self.layer2_importance(x)

        x = self.layer3(x)
        int_features.append(x)
        _,C,H,W = x.shape
        x = self.layer3_importance(x)

        x = self.layer4(x)
        int_features.append(x)
        _,C,H,W = x.shape
        x = self.layer4_importance(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, int_features

    def start_cal_importance(self):
        self._hook = [self.layer1_importance.register_backward_hook(gradhook),
                      self.layer2_importance.register_backward_hook(gradhook),
                      self.layer3_importance.register_backward_hook(gradhook),
                      self.layer4_importance.register_backward_hook(gradhook),
                      self.raw_features_importance.register_backward_hook(gradhook)]


    def reset_importance(self):
        self.layer1_importance.importance.zero_()
        self.layer2_importance.importance.zero_()
        self.layer3_importance.importance.zero_()
        self.layer4_importance.importance.zero_()
        self.raw_features_importance.importance.zero_()

    def normalize_importance(self):
        total_importance = torch.mean(self.layer1_importance.importance) + 1e-8
        self.layer1_importance.importance = self.layer1_importance.importance/total_importance
        total_importance = torch.mean(self.layer2_importance.importance) + 1e-8
        self.layer2_importance.importance = self.layer2_importance.importance/total_importance
        total_importance = torch.mean(self.layer3_importance.importance) + 1e-8
        self.layer3_importance.importance = self.layer3_importance.importance/total_importance
        total_importance = torch.mean(self.layer4_importance.importance) + 1e-8
        self.layer4_importance.importance = self.layer4_importance.importance/total_importance
        total_importance = torch.mean(self.raw_features_importance.importance) + 1e-8
        self.raw_features_importance.importance = self.raw_features_importance.importance/total_importance

    def stop_cal_importance(self):
        for hook in self._hook:
            hook.remove()
        self._hook = None


def resnet18(pretrained=False, remove_last_relu=False, num_segments=8, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], remove_last_relu, num_segments, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, remove_last_relu=False, num_segments=8, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], remove_last_relu, num_segments, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, remove_last_relu=False, num_segments=8, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], remove_last_relu, num_segments, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


