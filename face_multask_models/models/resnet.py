import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, landmarks_num=1000, img_size=224,dropout_factor = 1.):
        self.inplanes = 64
        self.dropout_factor = dropout_factor
        super(ResNet, self).__init__()
        # 26
        # 586 train_sequence
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # see this issue: https://github.com/xxradon/PytorchToCaffe/issues/16
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        assert img_size % 32 == 0
        pool_kernel = int(img_size / 32)
        self.avgpool = nn.AvgPool2d(pool_kernel, stride=1, ceil_mode=True)

        self.dropout1 = nn.Dropout(self.dropout_factor)
        self.dropout2 = nn.Dropout(0.8)
        self.dropout3 = nn.Dropout(0.8)

        self.dropout = nn.Dropout(self.dropout_factor)

        self.fc_landmarks_1 = nn.Linear(512 * block.expansion, 1024)
        self.fc_landmarks_2 = nn.Linear(1024, landmarks_num)

        self.fc_gender_1 = nn.Linear(512 * block.expansion, 64)
        self.fc_gender_2 = nn.Linear(64, 2)  # 男女

        self.fc_age_1 = nn.Linear(512 * block.expansion, 64)
        self.fc_age_2 = nn.Linear(64, 1)  # 年龄类别为1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
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
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

#         x = self.dropout(x)

        landmarks = self.fc_landmarks_1(x)
        landmarks = self.dropout1(landmarks)
        landmarks = self.fc_landmarks_2(landmarks)

        gender = self.fc_gender_1(x)
        gender = self.dropout2(gender)
        gender = self.fc_gender_2(gender)

        age = self.fc_age_1(x)
        age = self.dropout3(age)
        age = self.fc_age_2(age)


        return landmarks,gender,age # 多任务网络，返回关键点、性别、年龄


def load_model(model, pretrained_state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
                       k in model_dict and model_dict[k].size() == pretrained_state_dict[k].size()}
    model.load_state_dict(pretrained_dict, strict=False)
    if len(pretrained_dict) == 0:
        print("[INFO] No params were loaded ...")
    else:
        for k, v in pretrained_state_dict.items():
            if k in pretrained_dict:
                print("==>> Load {} {}".format(k, v.size()))
            else:
                print("[INFO] Skip {} {}".format(k, v.size()))
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print("Load pretrained model from {}".format(model_urls['resnet18']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        print("Load pretrained model from {}".format(model_urls['resnet34']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print("Load pretrained model from {}".format(model_urls['resnet50']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("Load pretrained model from {}".format(model_urls['resnet101']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        print("Load pretrained model from {}".format(model_urls['resnet152']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        model = load_model(model, pretrained_state_dict)
    return model

if __name__ == "__main__":
    input = torch.randn([32, 3, 256,256])
    model = resnet50(pretrained=False, landmarks_num=196, img_size=256)
    landmarks,gender,age = model(input)
    # landmarks shape:[bs,196]
    # gender shape:[bs,2]
    # age shape:[bs,1]
    print(landmarks.size(), gender.size(), age.size())
