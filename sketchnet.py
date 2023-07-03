import torch
import torch.nn as nn

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
        self.sketch_grads = {}

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