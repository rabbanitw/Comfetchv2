import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI


def conv3x3(in_planes, out_planes, stride=1, k=3, p=1, b=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=p, bias=b)


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
    def __init__(self, rank, depth, num_classes, sketch=True, same_sketch=True, cr=0.5, device=None, num_sketches=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # self.in_planes = 16
        self.cr = cr
        self.sketch = sketch
        self.rank = rank
        self.same_sketch = same_sketch
        self.device = device

        block, num_blocks = cfg(depth, sketch=self.sketch)

        if self.sketch:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
            # if devices using same sketching matrix
            if same_sketch:
                if self.rank == 0:
                    self.h = generate_sketch(self.in_planes, int(self.in_planes * self.cr), num_sketches)
                else:
                    self.h = None
                self.h = MPI.COMM_WORLD.bcast(self.h, root=0)
            else:
                self.h = generate_sketch(self.in_planes, int(self.in_planes * self.cr), num_sketches)

            # remove gradient tracking
            self.h.requires_grad = False
        else:
            self.conv1 = conv3x3(3, self.in_planes, k=7, stride=2, p=3, b=False)

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        '''
        self.layer1 = self._make_layer(block, 16, num_blocks[0], device, stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], device, stride=2)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], device, stride=2)
        self.layer4 = self._make_layer(block, 16, num_blocks[3], device, stride=2)
        self.linear = nn.Linear(16 * block.expansion, num_classes)
        '''
        self.layer1 = self._make_layer(block, 64, num_blocks[0], device, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], device, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], device, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], device, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, device, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            if self.sketch:
                layer = block(self.rank, self.in_planes, planes, self.cr, device, stride, same_sketch=self.same_sketch)
            else:
                layer = block(self.in_planes, planes, stride)
            layers.append(layer)
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.sketch:
            # load onto device
            self.h = self.h.to(self.device)
            out_sketch = sketch_mat(self.conv1.weight, self.h)
            out_conv = F.conv2d(x, out_sketch, padding=3, stride=2)
            out = unsketch_mat(out_conv, self.h.transpose(1, 2))
            out = F.relu(self.bn1(out))
        else:
            out = F.relu(self.bn1(self.conv1(x)))

        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class SketchBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, rank, in_planes, planes, cr, device, stride=1, num_sketches=1, same_sketch=True):
        super(SketchBasicBlock, self).__init__()
        self.stride = stride
        self.device = device
        self.rank = rank
        sketch_dim = int(planes*cr)

        # first piece
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if same_sketch:
            if self.rank == 0:
                self.h1 = generate_sketch(planes, sketch_dim, num_sketches)
                self.h2 = generate_sketch(planes, sketch_dim, num_sketches)
            else:
                self.h1 = None
                self.h2 = None
            self.h1 = MPI.COMM_WORLD.bcast(self.h1, root=0)
            self.h2 = MPI.COMM_WORLD.bcast(self.h2, root=0)
        else:
            self.h1 = generate_sketch(planes, sketch_dim, num_sketches)
            self.h2 = generate_sketch(planes, sketch_dim, num_sketches)

        # second piece
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # downstream
        self.shortcut = nn.Sequential()
        # if self.stride != 1:
        if self.stride != 1 or in_planes != planes:
            '''
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=self.stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )
            '''
            self.downsample = True
            self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=self.stride, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            # if devices using same sketching matrix
            if same_sketch:
                if self.rank == 0:
                    self.h3 = generate_sketch(planes, sketch_dim, num_sketches)
                else:
                    self.h3 = None
                self.h3 = MPI.COMM_WORLD.bcast(self.h3, root=0)
            else:
                self.h3 = generate_sketch(planes, sketch_dim, num_sketches)

            # remove gradient tracking
            self.h3.requires_grad = False
        else:
            self.downsample = False

        # remove gradient tracking
        self.h1.requires_grad = False
        self.h2.requires_grad = False

    def forward(self, x):

        # load onto device
        self.h1 = self.h1.to(self.device)
        self.h2 = self.h2.to(self.device)

        # forward first piece
        out_sketch1 = sketch_mat(self.conv1.weight, self.h1)
        out_conv1 = F.conv2d(x, out_sketch1, padding=1, stride=self.stride)
        out_unsketch1 = unsketch_mat(out_conv1, self.h1.transpose(1, 2))
        out1 = F.relu(self.bn1(out_unsketch1))

        # forward second piece
        out_sketch2 = sketch_mat(self.conv2.weight, self.h2)
        out_conv2 = F.conv2d(out1, out_sketch2, padding=1, stride=1)
        out_unsketch2 = unsketch_mat(out_conv2, self.h2.transpose(1, 2))
        out = self.bn2(out_unsketch2)

        if self.downsample:
            self.h3 = self.h3.to(self.device)
            out_sketch3 = sketch_mat(self.conv3.weight, self.h3)
            out_conv3 = F.conv2d(x, out_sketch3, stride=self.stride)
            out_unsketch3 = unsketch_mat(out_conv3, self.h3.transpose(1, 2))
            out += self.bn3(out_unsketch3)
        else:
            out += x

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
