import torch.nn as nn
import torch
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import random
import torch.nn.functional as F


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


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


class SimpleFPA(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPA, self).__init__()

        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

        # Global pooling branch
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        out = x_master + x_gpb

        return out


class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""
    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = SimpleFPA(B5_size, feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(B4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(B3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        B3, B4, B5 = inputs

        P5_x = self.P5_1(B5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(B4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(B3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""
    def __init__(self, channel_size=256):
        super(PyramidAttentions, self).__init__()

        self.A3_1 = SpatialGate(channel_size)
        self.A3_2 = ChannelGate(channel_size)

        self.A4_1 = SpatialGate(channel_size)
        self.A4_2 = ChannelGate(channel_size)

        self.A5_1 = SpatialGate(channel_size)
        self.A5_2 = ChannelGate(channel_size)

    def forward(self, inputs):
        F3, F4, F5 = inputs

        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A3 = A3_spatial*F3 + A3_channel*F3

        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_spatial*F4 + A4_channel*F4

        A5_spatial = self.A5_1(F5)
        A5_channel = self.A5_2(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_spatial*F5 + A5_channel*F5

        return [A3, A4, A5, A3_spatial, A4_spatial, A5_spatial]


class SpatialGate(nn.Module):
    """generation spatial attention mask"""
    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.ConvTranspose2d(out_channels,1,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)


class ChannelGate(nn.Module):
    """generation channel attention mask"""
    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels,out_channels//16,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(out_channels//16,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNet(nn.Module):
    """implementation of AP-CNN on ResNet"""
    def __init__(self, num_classes, lmda, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.lmda = lmda
        # self.process = process
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.num_classes == 200:
            hidden_num = 512
        else:
            hidden_num = 256

        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels,
                         self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        self.apn = PyramidAttentions(channel_size=256)

        self.cls5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),            
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls_concate = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(256*3),
            nn.Linear(256*3, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        
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

    def get_roi_crop_feat(self, x, roi_list, scale):
        """ROI guided refinement: ROI guided Zoom-in & ROI guided Dropblock"""
        n, c, x2_h, x2_w = x.size()
        # print('n', n)
        roi_3, roi_4, roi_5 = roi_list
        roi_all = torch.cat([roi_3, roi_4, roi_5], 0)
        x2_ret = []
        crop_info_all = []
        if self.training:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                roi_3_i = roi_3[roi_3[:, 0] == i] / scale
                roi_4_i = roi_4[roi_4[:, 0] == i] / scale
                # print('r:', roi_3_i)
                # print('r.size:', roi_3_i.size(0))
                # alway drop the roi with highest score
                mask_un = torch.ones(c, x2_h, x2_w).cuda()
                pro_rand = random.random()
                while pro_rand < 0.3 and roi_3_i.size(0) == 0:
                    pro_rand = random.random()
                if pro_rand < 0.3:
                    ind_rand = random.randint(0, roi_3_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_3_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_3_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                elif pro_rand < 0.6:
                    ind_rand = random.randint(0, roi_4_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_4_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_4_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                x2_drop = x[i] * mask_un
                x2_crop = x2_drop[:, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                # normalize
                scale_rate = c*(yy2_resize-yy1_resize)*(xx2_resize-xx1_resize) / torch.sum(mask_un[:, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()])
                x2_crop = x2_crop * scale_rate  

                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        else:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                x2_crop = x[i, :, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        return torch.cat(x2_ret, 0), crop_info_all

    def Concate(self, f3, f4, f5):
        f3 = nn.AdaptiveAvgPool2d(output_size=1)(f3)
        f5 = nn.AdaptiveAvgPool2d(output_size=1)(f5)
        f4 = nn.AdaptiveAvgPool2d(output_size=1)(f4)
        f_concate = torch.cat([f3, f4, f5], dim=1)
        return f_concate

    def forward(self, inputs, targets, process):
        # ResNet backbone with FC removed
        n, c, img_h, img_w = inputs.size()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # stage I
        f3, f4, f5 = self.fpn([x2, x3, x4])
        f3_att, f4_att, f5_att, a3, a4, a5 = self.apn([f3, f4, f5])
        # feature concat
        f_concate = self.Concate(f3, f4, f5) 
        out_concate = self.cls_concate(f_concate)
        loss_concate = self.criterion(out_concate, targets)
        out3 = self.cls3(f3_att)
        out4 = self.cls4(f4_att)
        out5 = self.cls5(f5_att)

        loss5 = self.criterion(out5, targets)

        loss = (1-self.lmda)*loss5 + self.lmda*loss_concate
        out = (1-self.lmda)*out5 + self.lmda*out_concate
        features = x4
        _, predicted = torch.max(out.data, 1)

        correct = predicted.eq(targets.data).cpu().sum().item()


        loss_ret = {'loss': loss}
        acc_ret = {'acc': correct}

        # attetion masks for visualizaton
        mask_cat = torch.cat([a3,
            F.interpolate(a4, a3.size()[2:]),
            F.interpolate(a5, a3.size()[2:])], 1)
        
        return loss_ret, acc_ret, mask_cat, out, features


def resnet50(num_classes, lmda, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, lmda, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
