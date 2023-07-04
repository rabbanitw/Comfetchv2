import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight, gain=math.sqrt(2))
        nn.init.constant(m.bias, 0)


def cfg(depth, sketch=True):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    if sketch:
        cf_dict = {
            '18': (SketchBasicBlock, [2, 2, 2, 2]),
            '34': (SketchBasicBlock, [3, 4, 6, 3])
        }
    else:
        cf_dict = {
            '18': (BasicBlock, [2, 2, 2, 2]),
            '34': (BasicBlock, [3, 4, 6, 3]),
            '50': (Bottleneck, [3, 4, 6, 3]),
            '101': (Bottleneck, [3, 4, 23, 3]),
            '152': (Bottleneck, [3, 8, 36, 3])
        }
    return cf_dict[str(depth)]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, depth, num_classes, sketch=True, cr=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.cr = cr
        self.sketch = sketch

        block, num_blocks = cfg(depth, sketch=self.sketch)

        self.conv1 = conv3x3(3, self.in_planes)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            if self.sketch:
                layer = block(self.in_planes, planes, self.cr, stride)
            else:
                layer = block(self.in_planes, planes, stride)
            layers.append(layer)
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class SketchBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cr, stride=1, num_sketches=1):
        super(SketchBasicBlock, self).__init__()
        self.stride = stride
        sketch_dim = int(planes*cr)

        # first piece
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=True)
        self.w1 = self.conv1.weight
        self.b1 = self.conv1.bias[:sketch_dim]

        self.h1 = generate_sketch(planes, sketch_dim, num_sketches)
        self.h1t = self.h1.transpose(1, 2)
        self.bn1 = nn.BatchNorm2d(planes)

        # second piece
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.w2 = self.conv2.weight
        self.b2 = self.conv2.bias[:sketch_dim]
        self.h2 = generate_sketch(planes, sketch_dim, num_sketches)
        self.h2t = self.h2.transpose(1, 2)
        self.bn2 = nn.BatchNorm2d(planes)

        # for now keep this as a normal convolutional process
        self.shortcut = nn.Sequential()
        if self.stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=self.stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # remove gradient tracking
        self.h1.requires_grad = False
        self.h1t.requires_grad = False
        self.h2.requires_grad = False
        self.h2t.requires_grad = False

    def forward(self, x):

        # forward first piece
        out_sketch1 = sketch_mat(self.w1, self.h1)
        out_conv1 = F.conv2d(x, out_sketch1, padding=1, stride=self.stride, bias=self.b1)
        out_unsketch1 = unsketch_mat(out_conv1, self.h1t)
        out1 = F.relu(self.bn1(out_unsketch1))

        # forward second piece
        out_sketch2 = sketch_mat(self.w2, self.h2)
        out_conv2 = F.conv2d(out1, out_sketch2, padding=1, stride=1, bias=self.b2)
        out_unsketch2 = unsketch_mat(out_conv2, self.h2t)
        out = self.bn2(out_unsketch2)

        # shortcut piece that I don't understand
        out += self.shortcut(x)
        return F.relu(out)


def sketch_mat(w: torch.Tensor, h):
    a, b, c, d = w.shape
    w = w.contiguous().view(a, -1)
    w = w.unsqueeze(0).repeat(h.shape[0], 1, 1)
    out = torch.matmul(h, w)
    return out.view(-1, b, c, d)


def unsketch_mat(w: torch.Tensor, ht):
    w = w.permute(1, 0, 2, 3)
    a, b, c, d = w.shape
    w = w.contiguous().view(a, -1)
    w = torch.stack(w.chunk(ht.shape[0]))
    w_unsk = torch.matmul(ht, w)
    w_unsk = torch.median(w_unsk, 0)[0]
    w_unsk = w_unsk.view(-1, b, c, d).permute(1, 0, 2, 3)
    return w_unsk


def generate_sketch(in_d, out_d, num_s=1):
    h = torch.zeros((num_s, out_d, in_d))
    hashed_indices = torch.randint(out_d, size=(in_d * num_s,))
    # determines random integers that are either 0 or 1, multiplies by 2 and subtracts by 1 to get 1 and -1
    rand_signs = torch.randint(2, size=(num_s, out_d, in_d)) * 2 - 1
    sketch_inds = torch.tile(torch.arange(num_s), (in_d,))
    column_inds = torch.repeat_interleave(torch.arange(in_d), num_s)
    h[sketch_inds, hashed_indices, column_inds] = 1
    sketch = h * rand_signs
    return sketch
    # return h.float(), rand_signs.float()
