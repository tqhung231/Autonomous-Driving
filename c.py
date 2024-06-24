import json
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter

# Load the entire checkpoint.
checkpoint = torch.load(
    "model/resnet18_bezierlanenet_tusimple_aug1b_20211109.pt", map_location="cpu"
)
print(checkpoint["model"].keys())

model = dict(
    name="BezierLaneNet",
    image_height=360,
    num_regression_parameters=8,  # 3 x 2 + 2 = 8 (Cubic Bezier Curve)
    # Inference parameters
    thresh=0.5,
    local_maximum_window_size=9,
    # Backbone (3-stage resnet (no dilation) + 2 extra dilated blocks)
    backbone_cfg=dict(
        name="predefined_resnet_backbone",
        backbone_name="resnet18",
        return_layer="layer3",
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
    ),
    reducer_cfg=None,  # No need here
    dilated_blocks_cfg=dict(
        name="predefined_dilated_blocks",
        in_channels=256,
        mid_channels=64,
        dilations=[4, 8],
    ),
    # Head, Fusion module
    feature_fusion_cfg=dict(name="FeatureFlipFusion", channels=256),
    head_cfg=dict(
        name="ConvProjection_1D", num_layers=2, in_channels=256, bias=True, k=3
    ),  # Just some transforms of feature, similar to FCOS heads, but shared between cls & reg branches
    # Auxiliary binary segmentation head (automatically discarded in eval() mode)
    aux_seg_head_cfg=dict(
        name="SimpleSegHead", in_channels=256, mid_channels=64, num_classes=1
    ),
)


def predefined_resnet_backbone(backbone_name, return_layer, **kwargs):
    backbone = resnet.__dict__[backbone_name](**kwargs)
    return_layers = parse_return_layers(return_layer)

    return IntermediateLayerGetter(backbone, return_layers=return_layers)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def predefined_dilated_blocks(in_channels, mid_channels, dilations):
    # As in YOLOF
    blocks = [
        MODELS.from_dict(
            dict(
                name="DilatedBottleneck",
                in_channels=in_channels,
                mid_channels=mid_channels,
                dilation=d,
            )
        )
        for d in dilations
    ]

    return nn.Sequential(*blocks)


class DilatedBottleneck(nn.Module):
    # Refactored from https://github.com/chensnathan/YOLOF/blob/master/yolof/modeling/encoder.py
    # Diff from typical ResNetV1.5 BottleNeck:
    # flexible expansion rate, forbids downsampling, relu immediately after last conv
    def __init__(self, in_channels=512, mid_channels=128, dilation=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out + identity

        return out


class FeatureFlipFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.proj2_conv = DCN_v2_Ref(channels, channels, kernel_size=(3, 3), padding=1)
        self.proj2_norm = nn.BatchNorm2d(channels)

    def forward(self, feature):
        # B x C x H x W
        flipped = feature.flip(-1)  # An auto-copy

        feature = self.proj1(feature)
        flipped = self.proj2_conv(flipped, feature)
        flipped = self.proj2_norm(flipped)

        return F.relu(feature + flipped)


class ConvProjection_1D(torch.nn.Module):
    # Projection based on line features (1D convs)
    def __init__(self, num_layers, in_channels, bias=True, k=3):
        # bias is set as True in FCOS
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i > 0 else in_channels,
                in_channels,
                kernel_size=k,
                bias=bias,
                padding=(k - 1) // 2,
            )
            for i in range(num_layers)
        )
        self.hidden_norms = nn.ModuleList(
            nn.BatchNorm1d(in_channels) for _ in range(num_layers)
        )

    def forward(self, x):
        for conv, norm in zip(self.hidden_layers, self.hidden_norms):
            x = F.relu(norm(conv(x)))

        return x


class SimpleSegHead(nn.Module):
    # Seg head (Generalized FCN head without dropout)
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class DCN_v2_Ref(ModulatedDeformConv2d):
    """A Encapsulation that acts as normal Conv
    layers. Modified from mmcv's DCNv2.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels * 2,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, ref):
        concat = torch.cat([x, ref], dim=1)
        out = self.conv_offset(concat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)
