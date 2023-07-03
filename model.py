from hmac import compare_digest
from pyparsing import nums
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1 or in_channels != out_channels:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out) + residual
        return out


class Net9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, image_dim=32, out_classes=10):
        super(Net9, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=256*(image_dim//16)**2, out_features=out_classes, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

# Adapted from torchvision implementation
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class Net18(nn.Module):
    def __init__(
        self,
        image_dim: int = 32,
        num_classes: int = 1000,
    ) -> None:
        super(Net18, self).__init__()

        norm_layer = nn.BatchNorm2d

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential(
            ResidualBlock(self.inplanes, 64, 3, 1, 1),
            ResidualBlock(64, 64, 3, 1, 1),

            ResidualBlock(64, 128, 3, 1, 1),
            ResidualBlock(128, 128, 3, 1, 1),
            
            ResidualBlock(128, 256, 3, 1, 1),
            ResidualBlock(256, 256, 3, 1, 1),
            
            ResidualBlock(256, 512, 3, 1, 1),
            ResidualBlock(512, 512, 3, 1, 1),
        )  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def generate_sketch(in_d, out_d, num_s=1):
    h = torch.zeros((num_s, out_d, in_d))
    hashed_indices = torch.randint(out_d, size=(in_d * num_s,))
    rand_signs = torch.randint(2, size=(num_s, in_d, 1)) * 2 - 1
    sketch_inds = torch.tile(torch.arange(num_s), (in_d,))
    column_inds = torch.repeat_interleave(torch.arange(in_d), num_s)
    h[sketch_inds, hashed_indices, column_inds] = 1
    return h.float(), rand_signs.float()

def sketch_mat(w: torch.Tensor, h, s):
    a, b, c, d = w.shape
    w = w.contiguous().view(a, -1)
    w = w.unsqueeze(0).repeat(s.shape[0], 1, 1)
    out = h @ (w * s)
    return out.view(-1, b, c, d)

def unsketch_mat(w: torch.Tensor, h, s):
    w = w.permute(1, 0, 2, 3)
    a, b, c, d = w.shape
    w = w.contiguous().view(a, -1)
    w = torch.stack(w.chunk(h.shape[0]))
    w_unsk = (h.transpose(1, 2) @ w) * s
    w_unsk = torch.median(w_unsk, 0)[0]
    w_unsk = w_unsk.view(-1, b, c, d).permute(1, 0, 2, 3)
    return w_unsk

class SketchResidualBlock(nn.Module):
    def __init__(self, comp_rate, num, in_channels, out_channels, kernel_size, padding, stride, num_sketches):
        super(SketchResidualBlock, self).__init__()
        stride = 1
        conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.w1 = conv_res1.weight
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.w2 = conv_res2.weight
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=False)
        self.num = num
        self.h, self.s = generate_sketch(out_channels, out_channels//comp_rate, 5)
        self.sketch_grads = {}
 
    def forward(self, x):
        residual = x
        #out = self.conv_res1(x)
        w1_sk = sketch_mat(self.w1, self.h.to(x.device), self.s.to(x.device))
        out = F.conv2d(x, w1_sk, padding=1, stride=1)
        out = unsketch_mat(out, self.h.to(x.device), self.s.to(x.device))
        out = self.relu(self.conv_res1_bn(out))
        #out = self.conv_res2(x)
        w2_sk = sketch_mat(self.w2, self.h.to(x.device), self.s.to(x.device))
        out = F.conv2d(out, w2_sk, padding=1, stride=1)
        out = unsketch_mat(out, self.h.to(x.device), self.s.to(x.device))
        out = self.relu(self.conv_res2_bn(out)) + residual

        return out

class SketchNet(nn.Module):
    def __init__(self, comp_rate, image_dim=32, out_classes=10, num_sketches=1):
        super(SketchNet, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.w1 = conv1.weight
        conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.w2 = conv2.weight
        conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.w3 = conv3.weight
        conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.w4 = conv4.weight

        self.res_block1 = SketchResidualBlock(
            comp_rate, num=1, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
            num_sketches=num_sketches
        )
        self.res_block2 = SketchResidualBlock(
            comp_rate, num=2, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
            num_sketches=num_sketches
        )
        
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.9)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=256*(image_dim//16)**2, out_features=out_classes, bias=True)

        self.h1, self.s1 = generate_sketch(64, 64//comp_rate, num_sketches)
        self.h2, self.s2 = generate_sketch(128, 128//comp_rate, num_sketches)
        self.h3, self.s3 = generate_sketch(256, 256//comp_rate, num_sketches)
        
        self.sketch_grads = {}
        self.sketches = [(self.h1, self.s1),(self.h2, self.s2),(self.h3, self.s3),(self.h3, self.s3),
                        (self.res_block1.h, self.res_block1.s), (self.res_block1.h, self.res_block1.s),
                        (self.res_block2.h, self.res_block2.s), (self.res_block2.h, self.res_block2.s)]

    def forward(self, x):
        w1_sk = sketch_mat(self.w1, self.h1.to(x.device), self.s1.to(x.device))
        out = F.conv2d(x, w1_sk, padding=1, stride=1)
        out = unsketch_mat(out, self.h1.to(x.device), self.s1.to(x.device))
        out = self.relu(self.bn1(out))

        w2_sk = sketch_mat(self.w2, self.h2.to(x.device), self.s2.to(x.device))
        out = F.conv2d(out, w2_sk, padding=1, stride=1)
        out = unsketch_mat(out, self.h2.to(x.device), self.s2.to(x.device))
        out = self.pool(self.relu(self.bn2(out)))

        out = self.res_block1(out)
        
        w3_sk = sketch_mat(self.w3, self.h3.to(x.device), self.s3.to(x.device))
        out = F.conv2d(out, w3_sk, padding=1, stride=1)
        out = unsketch_mat(out, self.h3.to(x.device), self.s3.to(x.device))
        out = self.pool(self.relu(self.bn3(out)))

        w4_sk = sketch_mat(self.w4, self.h3.to(x.device), self.s3.to(x.device))
        out = F.conv2d(out, w4_sk, padding=1, stride=1)
        out = unsketch_mat(out, self.h3.to(x.device), self.s3.to(x.device))
        out = self.pool(self.relu(self.bn4(out)))

        out = self.pool(self.res_block2(out))

        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        
        return out

class SketchNet18(nn.Module):
    def __init__(
        self, comp_rate: float, image_dim: int = 32, num_classes: int = 1000, num_sketches: int = 1
    ) -> None:
        super(SketchNet18, self).__init__()

        norm_layer = nn.BatchNorm2d
        # layer_counts = [2, 2, 2, 2]
        self.inplanes = 64
        
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.w1 = conv1.weight
        self.h1, self.s1 = generate_sketch(self.inplanes, self.inplanes // comp_rate, num_sketches)
        conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.w2 = conv2.weight
        self.h2, self.s2 = generate_sketch(128, 128 // comp_rate, num_sketches)
        conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.w3 = conv3.weight
        self.h3, self.s3 = generate_sketch(256, 256 // comp_rate, num_sketches)
        conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.w4 = conv4.weight
        self.h4, self.s4 = generate_sketch(512, 512 // comp_rate, num_sketches)
          
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(256)
        self.bn4 = norm_layer(512)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer = nn.ModuleList([SketchResidualBlock(comp_rate, 1, 64, 64, 3, 1, 1)])

        self.layers = nn.ModuleList([
            SketchResidualBlock(comp_rate, 1, 64, 64, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 64, 64, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 128, 128, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 128, 128, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 256, 256, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 256, 256, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 512, 512, 3, 1, 1, num_sketches),
            SketchResidualBlock(comp_rate, 1, 512, 512, 3, 1, 1, num_sketches),
        ])

        self.sketches = [(self.h1, self.s1), (self.h2, self.s2), (self.h3, self.s3), (self.h4, self.s4), 
            (self.layers[0].h, self.layers[0].s), (self.layers[0].h, self.layers[0].s), 
            (self.layers[1].h, self.layers[1].s), (self.layers[1].h, self.layers[1].s),
            (self.layers[2].h, self.layers[2].s), (self.layers[2].h, self.layers[2].s), 
            (self.layers[3].h, self.layers[3].s), (self.layers[3].h, self.layers[3].s),
            (self.layers[4].h, self.layers[4].s), (self.layers[4].h, self.layers[4].s),
            (self.layers[5].h, self.layers[5].s), (self.layers[5].h, self.layers[5].s),
            (self.layers[6].h, self.layers[6].s), (self.layers[6].h, self.layers[6].s),
            (self.layers[7].h, self.layers[7].s), (self.layers[7].h, self.layers[7].s)          
        ]
        # self.sketches = [
        #     (self.h1, self.s1),
        #     (self.h2, self.s2),# (self.layers[2].h, self.layers[2].s), (self.layers[3].h, self.layers[3].s),
        #     (self.h3, self.s3), #(self.layers[4].h, self.layers[4].s), (self.layers[5].h, self.layers[5].s), 
        #     (self.h4, self.s4), #(self.layers[6].h, self.layers[6].s), (self.layers[7].h, self.layers[7].s),
        #     (self.layer[0].h, self.layer[0].s),            
        #     (self.layer[0].h, self.layer[0].s),            
        # ]
        self.sketch_grads = {}

        # for i in range(4):
        #     self.layers[i] = nn.Sequential(*self.layers[i]) 
        # self.layers = nn.ModuleList(self.layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_sk = sketch_mat(self.w1.to(x.device), self.h1.to(x.device), self.s1.to(x.device))
        x = F.conv2d(x, w1_sk, padding=1, stride=1)
        x = unsketch_mat(x, self.h1.to(x.device), self.s1.to(x.device))
        x = self.relu(self.bn1(x))

        x = self.layers[0](x)
        x = self.layers[1](x)

        w2_sk = sketch_mat(self.w2, self.h2.to(x.device), self.s2.to(x.device))
        x = F.conv2d(x, w2_sk, padding=1, stride=1)
        x = unsketch_mat(x, self.h2.to(x.device), self.s2.to(x.device))
        x = self.relu(self.bn2(x))

        x = self.layers[2](x)
        x = self.layers[3](x)

        w3_sk = sketch_mat(self.w3, self.h3.to(x.device), self.s3.to(x.device))
        x = F.conv2d(x, w3_sk, padding=1, stride=1)
        x = unsketch_mat(x, self.h3.to(x.device), self.s3.to(x.device))
        x = self.relu(self.bn3(x))

        x = self.layers[4](x)
        x = self.layers[5](x)

        w4_sk = sketch_mat(self.w4, self.h4.to(x.device), self.s4.to(x.device))
        x = F.conv2d(x, w4_sk, padding=1, stride=1)
        x = unsketch_mat(x, self.h4.to(x.device), self.s4.to(x.device))
        x = self.relu(self.bn4(x))
       
        x = self.layers[6](x)
        x = self.layers[7](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Sketch3ResidualBlock(nn.Module):
    def __init__(self, comp_rate, num, in_channels, out_channels, kernel_size, padding, stride):
        super(Sketch3ResidualBlock, self).__init__()
        stride = 1
        conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.w1 = conv_res1.weight
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.w2 = conv_res2.weight
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=False)
        self.num = num
        
        self.hs, self.ss = [], []
        for _ in range(3):
            h1, s1 = generate_sketch(out_channels, out_channels//comp_rate)
            self.hs.append(h1); self.ss.append(s1)

    def forward(self, x):
        residual = x
        #out = self.conv_res1(x)
        c_outs = []
        for h, s in zip(self.hs, self.ss):
            w_sk = sketch_mat(self.w1, h.to(x.device), s.to(x.device))
            out = F.conv2d(x, w_sk, padding=1, stride=1)
            c_outs.append(unsketch_mat(out, h.to(x.device), s.to(x.device)))
        out = torch.median(torch.stack(c_outs, dim=0), dim=0).values
        out = self.relu(self.conv_res1_bn(out))
        #out = self.conv_res2(x)
        c_outs = []
        for h, s in zip(self.hs, self.ss):
            w_sk = sketch_mat(self.w2, h.to(x.device), s.to(x.device))
            out = F.conv2d(x, w_sk, padding=1, stride=1)
            c_outs.append(unsketch_mat(out, h.to(x.device), s.to(x.device)))
        out = torch.median(torch.stack(c_outs, dim=0), dim=0).values
        out = self.relu(self.conv_res2_bn(out))
        out += residual

        return out

class Sketch3Net(nn.Module):
    def __init__(self, comp_rate, image_dim=32, out_classes=10):
        super(Sketch3Net, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.w1 = conv1.weight
        conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.w2 = conv2.weight
        conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.w3 = conv3.weight
        conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.w4 = conv4.weight

        self.res_block1 = Sketch3ResidualBlock(comp_rate, num=1, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.res_block2 = Sketch3ResidualBlock(comp_rate, num=2, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.9)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=256*(image_dim//16)**2, out_features=out_classes, bias=True)

        self.h1s, self.s1s = [], []
        self.h2s, self.s2s = [], []
        self.h3s, self.s3s = [], []
        for _ in range(3):
            h1, s1 = generate_sketch(64, 64//comp_rate)
            self.h1s.append(h1); self.s1s.append(s1)
            h2, s2 = generate_sketch(128, 128//comp_rate)
            self.h2s.append(h2); self.s2s.append(s2)
            h3, s3 = generate_sketch(256, 256//comp_rate)
            self.h3s.append(h3); self.s3s.append(s3)

    def forward(self, x):
        c_outs = []
        for h1, s1 in zip(self.h1s, self.s1s):
            w1_sk = sketch_mat(self.w1, h1.to(x.device), s1.to(x.device))
            out = F.conv2d(x, w1_sk, padding=1, stride=1)
            c_outs.append(unsketch_mat(out, h1.to(x.device), s1.to(x.device)))
        out = torch.median(torch.stack(c_outs, dim=0), dim=0).values
        out = self.relu(self.bn1(out))
        
        c_outs = []
        for h, s in zip(self.h2s, self.s2s):
            w_sk = sketch_mat(self.w2, h.to(x.device), s.to(x.device))
            out = F.conv2d(x, w_sk, padding=1, stride=1)
            c_outs.append(unsketch_mat(out, h.to(x.device), s.to(x.device)))
        out = torch.median(torch.stack(c_outs, dim=0), dim=0).values
        out = self.pool(self.relu(self.bn2(out)))

        out = self.res_block1(out)
        
        c_outs = []
        for h, s in zip(self.h3s, self.s3s):
            w_sk = sketch_mat(self.w3, h.to(x.device), s.to(x.device))
            out = F.conv2d(x, w_sk, padding=1, stride=1)
            c_outs.append(unsketch_mat(out, h.to(x.device), s.to(x.device)))
        out = torch.median(torch.stack(c_outs, dim=0), dim=0).values
        out = self.pool(self.relu(self.bn3(out)))

        c_outs = []
        for h, s in zip(self.h3s, self.s3s):
            w_sk = sketch_mat(self.w4, h.to(x.device), s.to(x.device))
            out = F.conv2d(x, w_sk, padding=1, stride=1)
            c_outs.append(unsketch_mat(out, h.to(x.device), s.to(x.device)))
        out = torch.median(torch.stack(c_outs, dim=0), dim=0).values
        out = self.pool(self.relu(self.bn4(out)))

        out = self.pool(self.res_block2(out))

        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        
        return out

