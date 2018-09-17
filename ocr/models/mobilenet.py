import math
import torch
import torch.nn as nn


def _conv_bn(inp, out, stride):
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU6(inplace=True),
    )


def _conv_1x1_bn(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, out, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, out, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == out

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(inp, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # point-wise
                nn.Conv2d(hidden_dim, out, 1, stride=1, padding=0, bias=False),
                nn.ReLU6(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                # point-wise
                nn.Conv2d(inp, hidden_dim, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # point-wise
                nn.Conv2d(hidden_dim, out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def forward(self, input):
        if self.use_res_connect:
            return input + self.conv(input)
        else:
            return self.conv(input)


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 32
        inverted_residual_config = [
            # t (expand ratio), channel, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.conv1 = _conv_bn(3, input_channel, 2)
        for i, (t, c, n, s) in enumerate(inverted_residual_config):
            output_channel = c
            layers = []
            for j in range(n):
                if j == 0:
                    layers.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
            setattr(self, f'block{i}', nn.Sequential(*layers))
        self.last_channel = 1280
        self.conv2 = _conv_1x1_bn(input_channel, self.last_channel)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block0(x)
        p1 = x = self.block1(x)
        p2 = x = self.block2(x)
        x = self.block3(x)
        p3 = x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return p1, p2, p3, self.conv2(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = MobileNetV2()

        def upsampling(inp, out):
            return nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear'),
            )

        self.up1 = upsampling(self.mobilenet.last_channel, 96)
        self.up2 = upsampling(96, 32)
        self.up3 = upsampling(32, 24)

    def forward(self, image):
        p1, p2, p3, p4 = self.mobilenet(image)
        x = self.up1(p4) + p3
        x = self.up2(x) + p2
        x = self.up3(x) + p1
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    net = MobileNetV2Backbone().to(device)
    images = torch.randn(4, 3, 224, 256).to(device)
    output = net(images)
    print(output.size())
