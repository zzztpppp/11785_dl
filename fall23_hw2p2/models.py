import math

import torch
import torch.nn.functional as nf
from torch import nn


class Network(nn.Module):
    """
    The Very Low early deadline architecture is a 5-layer CNN. Keep in mind the parameter limit is 21M.

    The first Conv layer has 64 channels, kernel size 7, and stride 4.
    The next three have 128, 256, 512 and 1024 channels. Each have kernel size 3 and stride 2.

    Think about strided convolutions from the lecture, as convolutioin with stride= 1 and downsampling.
    For stride 1 convolution, what padding do you need for preserving the spatial resolution?
    (Hint => padding = kernel_size // 2) - Why?)

    Each Conv layer is accompanied by a Batchnorm and ReLU layer.
    Finally, you want to average pool over the spatial dimensions to reduce them to 1 x 1. Use AdaptiveAvgPool2d.
    Then, remove (Flatten?) these trivial 1x1 dimensions away.
    Look through https://pytorch.org/docs/stable/nn.html

    TODO: Fill out the model definition below!

    Why does a very simple network have 4 convolutions?
    Input images are 224x224. Note that each of these convolutions downsample.
    Downsampling 2x effectively doubles the receptive field, increasing the spatial
    region each pixel extracts features from. Downsampling 32x is standard
    for most image models.

    Why does a very simple network have high channel sizes?
    Every time you downsample 2x, you do 4x less computation (at same channel size).
    To maintain the same level of computation, you 2x increase # of channels, which
    increases computation by 4x. So, balances out to same computation.
    Another intuition is - as you downsample, you lose spatial information. We want
    to preserve some of it in the channel dimension.
    """

    def __init__(self, num_classes=7001):
        super().__init__()

        self.backbone = torch.nn.Sequential(

            # TODO

        )

        # self.cls_layer = #TODO

    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out


def l2_norm(inputs, dim):
    norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
    return inputs / norm


class ConvNextBlock(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_size,
                out_channels=channel_size,
                groups=channel_size,
                kernel_size=7,
                padding=3,
            ),
            LayerNorm(channel_size, data_format="channels_first"),
            # Point-wise convolution by linear layers
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=channel_size * 4, out_channels=channel_size, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers.forward(x)


class ConvNextBlock2(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        inputs = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = inputs + x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return nf.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextT(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self._channel_sizes = [96, 192, 384, 768]
        self._stage_depths = [2, 2, 3, 2]
        # 4X down-sample
        self._down_sample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                kernel_size=4,
                stride=4,
                in_channels=input_channels,
                out_channels=self._channel_sizes[0]
            ),
            LayerNorm(normalized_shape=self._channel_sizes[0], data_format="channels_first")
        )
        self._down_sample_layers.append(stem)
        for i in range(3):
            down_sample = nn.Sequential(
                LayerNorm(self._channel_sizes[i], data_format="channels_first"),
                nn.Conv2d(
                    in_channels=self._channel_sizes[i],
                    out_channels=self._channel_sizes[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )
            self._down_sample_layers.append(down_sample)
        self._stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNextBlock2(dim=self._channel_sizes[i]) for _ in range(self._stage_depths[i])]
            )
            self._stages.append(stage)

        # Feature norm
        self._norm = nn.LayerNorm(self._channel_sizes[-1])

        # Classifier
        self._cls = nn.Linear(self._channel_sizes[-1], num_classes)

    def forward_feature(self, x):
        for i in range(4):
            x = self._down_sample_layers[i].forward(x)
            x = self._stages[i].forward(x)

        # Average pooling, from (N, C, H, W) -> (N, C)
        return self._norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_feature(x)
        return self._cls(x)


class ResidualBlock(nn.Module):
    """
    A size customizable residual block from recitation slides.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(output_channels),
        )
        # Keep the input size and output size match
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.layers(x)
        shortcut = self.shortcut(x)
        return nf.relu(out + shortcut)


class ResNet34(nn.Module):

    def __init__(self, n_classes, cf_dropout=False):
        super(ResNet34, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=6),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3, 2),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 128, 3),  # (128, 7, 7)
            ResidualBlock(128, 256, 3, 2),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 256, 3),  # (256, 3, 3)
            ResidualBlock(256, 512, 3, 2),  # (512, 1, 1)
            ResidualBlock(512, 512, 3),  # (512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
        )
        self._cf_dropout = cf_dropout
        self.class_out = nn.Linear(1000, n_classes)

    def forward(self, x):
        net_out = self.layers(x)
        if self._cf_dropout:
            net_out = nf.dropout(net_out)
        return self.class_out(net_out)

    def forward_feat(self, x):
        output = self.layers(x)
        return output


class ResNet18(nn.Module):
    """
    An 18-layer ResNet from recitation slides.
    """
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),  # (64, 30, 30)
            nn.MaxPool2d(3, 2),      # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 64, 3),  # (64, 14, 14)
            ResidualBlock(64, 128, 3, True),  # (128, 14, 14)
            ResidualBlock(128, 128, 3),  # (128, 14, 14)
            ResidualBlock(128, 256, 3, True),  # (256, 14, 14)
            ResidualBlock(256, 256, 3),  # (256, 14, 14)
            ResidualBlock(256, 512, 3, True),  # (512, 14, 14)
            ResidualBlock(512, 512, 3),  # (512, 14, 14)
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = nf.linear(nf.normalize(input), nf.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
