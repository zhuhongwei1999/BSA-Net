import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Src.backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from Src.module.BasicConv2d import BasicConv2d
from Src.module.RMFE import RF2B
from Src.module.SA import sa_layer
from Src.module.Fusion import Fusion, aggregation
from Src.module.BG import Spade
from Src.module.SEA import MSCA

class BSANet(nn.Module):
    def __init__(self):
        super(BSANet, self).__init__()
        self.backbone = res2net50_v1b_26w_4s()
        self.RME1 = RF2B(256, 64)
        self.RME2 = RF2B(512, 64)
        self.RME3 = RF2B(1024, 64)
        self.RME4 = RF2B(2048, 64)

        self.edge_conv1 = BasicConv2d(256, 64, kernel_size=3, padding=1)
        self.edge_conv2 = BasicConv2d(512, 64, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(1024, 64, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(2048, 64, kernel_size=3, padding=1)

        self.edge_conv_cat = BasicConv2d(64 * 4, 64, kernel_size=3, padding=1)
        self.edge_linear = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.agg = aggregation(64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.upsample4_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.upsample8_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.conv1 = BasicConv2d(256, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.spade1 = Spade(64, 64)
        self.spade2 = Spade(64, 64)
        self.spade3 = Spade(64, 64)
        self.spade4 = Spade(64, 64)

        self.spade5 = Spade(64, 64)
        self.spade6 = Spade(64, 64)
        self.spade7 = Spade(64, 64)

        self.fusion1 = Fusion(64)
        self.fusion2 = Fusion(64)
        self.fusion3 = Fusion(64)

        self.msca1 = MSCA()
        self.msca2 = MSCA()
        self.msca3 = MSCA()
        self.msca4 = MSCA()

        self.SA1 = sa_layer(64)
        self.SA2 = sa_layer(64)
        self.SA3 = sa_layer(64)
        self.SA4 = sa_layer(64)

        self.ra11_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra21_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra31_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra41_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra1_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra4_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.rra1_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.rra2_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.rra3_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.rra4_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.refra1_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.refra2_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.refra3_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.refra4_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearrr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearrr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearrr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearrr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr6 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr7 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr8 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x):
        image_shape = x.size()[2:]
        x_backbone = self.backbone.conv1(x)
        x_backbone = self.backbone.bn1(x_backbone)
        x_backbone = self.backbone.relu(x_backbone)
        x_backbone = self.backbone.maxpool(x_backbone)

        layer1 = self.backbone.layer1(x_backbone)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        layer1_edge = self.edge_conv1(layer1)
        layer2_edge = self.edge_conv2(layer2)
        layer3_edge = self.edge_conv3(layer3)
        layer4_edge = self.edge_conv4(layer4)

        layer2_edge = self.upsample2(layer2_edge)
        layer3_edge = self.upsample4(layer3_edge)
        layer4_edge = self.upsample8(layer4_edge)

        edge_cat = self.edge_conv_cat(torch.cat((layer1_edge, layer2_edge, layer3_edge, layer4_edge), dim=1))
        edge_map = self.edge_linear(edge_cat)
        edge_out = F.interpolate(edge_map, size=image_shape, mode='bilinear')

        rme1, rme2, rme3, rme4 = self.RME1(layer1), self.RME2(layer2), self.RME3(layer3), self.RME4(layer4)

        map_4 = self.linearr4(rme4)
        out_4 = F.interpolate(map_4, size=image_shape, mode='bilinear')

        ra_4 = F.interpolate(rme4, size=rme3.size()[2:], mode='bilinear')
        ra_4 = self.ra41_conv(ra_4)
        ra_4 = 1 - torch.sigmoid(ra_4)
        ra_4_weight = ra_4.expand(-1, rme3.size()[1], -1, -1)
        rra_4_weight = 1 - ra_4_weight
        ra_4_out = ra_4_weight * rme3
        rra_4_out = rra_4_weight * rme3
        ra_4_out = self.ra4_conv(ra_4_out)
        rra_4_out = self.rra4_conv(rra_4_out)
        map_3 = self.linearr3(ra_4_out)
        out_3 = F.interpolate(map_3, size=image_shape, mode='bilinear')

        ra_3 = F.interpolate(ra_4_out, size=rme2.size()[2:], mode='bilinear')
        ra_3 = self.ra31_conv(ra_3)
        ra_3 = 1 - torch.sigmoid(ra_3)
        ra_3_weight = ra_3.expand(-1, rme2.size()[1], -1, -1)
        rra_3_weight = 1 - ra_3_weight
        ra_3_out = ra_3_weight * rme2
        rra_3_out = rra_3_weight * rme2
        ra_3_out = self.ra3_conv(ra_3_out)
        rra_3_out = self.rra3_conv(rra_3_out)
        map_2 = self.linearr2(ra_3_out)
        out_2 = F.interpolate(map_2, size=image_shape, mode='bilinear')

        ra_2 = F.interpolate(ra_3_out, size=rme1.size()[2:], mode='bilinear')
        ra_2 = self.ra21_conv(ra_2)
        ra_2 = 1 - torch.sigmoid(ra_2)
        ra_2_weight = ra_2.expand(-1, rme1.size()[1], -1, -1)
        rra_2_weight = 1 - ra_2_weight
        ra_2_out = ra_2_weight * rme1
        rra_2_out = rra_2_weight * rme1
        ra_2_out = self.ra2_conv(ra_2_out)
        rra_2_out = self.rra2_conv(rra_2_out)
        map_1 = self.linearr1(ra_2_out)
        out_1 = F.interpolate(map_1, size=image_shape, mode='bilinear')

        guider1 = ra_2_out
        guider2 = F.interpolate(ra_3_out, scale_factor=2, mode='bilinear')
        guider3 = F.interpolate(ra_4_out, scale_factor=4, mode='bilinear')
        guider4 = F.interpolate(rme4, scale_factor=8, mode='bilinear')

        guider5 = rra_2_out
        guider6 = F.interpolate(rra_3_out, scale_factor=2, mode='bilinear')
        guider7 = F.interpolate(rra_4_out, scale_factor=4, mode='bilinear')

        weight1 = self.msca1(guider1 + guider5)
        weight2 = self.msca2(guider2 + guider6)
        weight3 = self.msca3(guider3 + guider7)

        spade4 = self.spade4(guider4, edge_map)
        spade3 = self.spade3(guider3 * weight3, edge_map)
        spade2 = self.spade2(guider2 * weight2, edge_map)
        spade1 = self.spade1(guider1 * weight1, edge_map)

        spade5 = self.spade5(guider5 * (1 - weight1), edge_map)
        spade6 = self.spade6(guider6 * (1 - weight2), edge_map)
        spade7 = self.spade7(guider7 * (1 - weight3), edge_map)

        SA_4 = self.SA4(spade4)
        SA_3 = self.SA3(spade3 + spade7)
        SA_2 = self.SA2(spade2 + spade6)
        SA_1 = self.SA1(spade1 + spade5)

        SA_4 = SA_4
        SA_3 = self.fusion3(SA_3, SA_4)
        SA_2 = self.fusion2(SA_2, SA_3)
        SA_1 = self.fusion1(SA_1, SA_2)

        map_24 = self.linearr8(SA_4)
        map_23 = self.linearr7(SA_3) + map_24
        map_22 = self.linearr6(SA_2) + map_23
        map_21 = self.linearr5(SA_1) + map_22

        out_21 = F.interpolate(map_21, size=image_shape, mode='bilinear')
        out_22 = F.interpolate(map_22, size=image_shape, mode='bilinear')
        out_23 = F.interpolate(map_23, size=image_shape, mode='bilinear')
        out_24 = F.interpolate(map_24, size=image_shape, mode='bilinear')

        return out_1, out_2, out_3, out_4, out_21, out_22, out_23, out_24, edge_out

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            m.initialize()