#**coding=utf-8**
import torch.nn as nn
import torch.functional
import math
#IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks. Ke Sun, Mingjie Li, Dong Liu, and Jingdong Wang. arXiv preprint arXIV:1806.00178 (2017)
#
def conv_bn(inp, oup, stride ):
    return nn.Sequential(
        nn.Conv2d(inp, oup,kernel_size= 3, stride= stride, padding= 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size = 1, stride= 1, padding= 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        #直接就是mxnet实现的permutation操作
        # def permutation(data, groups):
            #举例说明：当groups = 2时，输入：nx144x56x56
        #     data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
        #            输出：nx2x72x56x56
        #     data = mx.sym.swapaxes(data, 1, 2)
        #            输出：nx72x2x56x56
        #     data = mx.sym.reshape(data, shape=(0, -3, -2))
        #            输出：nx144x56x56
        #     return data
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return output


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio,kernel_size = 1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            #permutation
            PermutationBlock(groups=2),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size =3, stride= stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size =1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(oup),
            # permutation
            PermutationBlock(groups= int(round((oup/2)))),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class IGCV3(nn.Module):
    def __init__(self,args):
        super(IGCV3, self).__init__()
        # 配置某些block的stride，满足downsampling的需求
        s1, s2 = 2, 2
        if args.downsampling == 16:
            s1, s2 = 2, 1
        elif args.downsampling == 8:
            s1, s2 = 1, 1

        '''
                network_settings网络的相关配置，从该参数可以看出，Mobile-Net由9个部分组成,
                姑且叫做Mobile block。
                network_settings中:
                't'表示Inverted Residuals的扩征系数
                'c'表示该block输出的通道数
                ‘n’表示当前block由几个残差单元组成
                's'表示当前block的stride
                '''
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 4, s2],
            [6, 32, 6, 2],
            [6, 64, 8, 2],
            [6, 96, 6, 1],
            [6, 160, 6, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert args.img_height % 32 == 0
        input_channel = int(32 * args.width_multiplier)
        self.last_channel = int(1280 * args.width_multiplier) if args.width_multiplier > 1.0 else 1280
        #第一层，
        self.features = [conv_bn(inp =3, oup =input_channel, stride = s1)]
        #中间block，一共7个,
        #  Layers from 1 to 7
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * args.width_multiplier)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(kernel_size = (args.img_height // args.downsampling, args.img_width // args.downsampling)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel,args.num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #     nn.init.xavier_normal(m.weight)
            #     if m.bias is not None:
            #         nn.init.constant(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant(m.weight, 1)
            #     nn.init.constant(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

