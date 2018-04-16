import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 2 * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != 2 * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, 2 * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(2 * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FeatureExtractNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 1, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.latlayer1 = nn.Conv2d(2 * 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1 * 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        self.toplayer1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, planes, block, stride):
        strides = [stride] + [1] * (block - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * 2
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3


def main():
    import torch
    from torch.autograd import Variable
    import torchvision.transforms as transforms
    from data import ListDataset

    use_cuda = False
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    trainset = ListDataset(root='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=1, collate_fn=trainset.collate_fn)
    net = FeatureExtractNet()
    if use_cuda:
        net.cuda()
    for batch_idx, (inputs, loc_targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
        inputs = Variable(inputs)
        loc_targets = Variable(loc_targets)

        feature_maps = net(inputs)
        print(feature_maps.size())
        break
