import torch
import torch.nn as nn
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN
from functools import partial

import models
from modules import DeeplabV3, DeeplabV2


def make_model(opts):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)

    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head_channels = 256
    if opts.deeplab == 'v3':
        head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                         out_stride=opts.output_stride, pooling_size=opts.pooling)
    elif opts.deeplab == 'v2':
        head = DeeplabV2(body.out_channels, head_channels, norm_act=norm,
                         out_stride=opts.output_stride)
    else:
        raise NotImplementedError("Specify a correct head.")

    model = SegmentationModule(body, head, head_channels, opts.num_classes)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class TestSegmentationModule(nn.Module):
    _IGNORE_INDEX = 255

    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return probs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(x.size(0), x.size(2), x.size(3), dtype=torch.long)
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, body, head, head_channels, classes, fusion_mode="mean"):
        super(TestSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes
        if fusion_mode == "mean":
            self.fusion_cls = TestSegmentationModule._MeanFusion
        elif fusion_mode == "voting":
            self.fusion_cls = TestSegmentationModule._VotingFusion
        elif fusion_mode == "max":
            self.fusion_cls = TestSegmentationModule._MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [round(s * scale) for s in x.shape[-2:]]
            x_up = functional.interpolate(x, size=scaled_size, mode="bilinear", align_corners=False)
        else:
            x_up = x

        x_up = self.body(x_up)
        x_up = self.head(x_up)
        sem_logits = self.cls(x_up)

        del x_up
        return sem_logits

    def forward(self, x, scales=None, do_flip=False):
        if scales is None:
            scales = [1.]

        out_size = x.shape[-2:]
        fusion = self.fusion_cls(x, self.classes)

        for scale in scales:
            # Main orientation
            sem_logits = self._network(x, scale)
            sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            fusion.update(sem_logits)

            # Flipped orientation
            if do_flip:
                # Main orientation
                sem_logits = self._network(flip(x, -1), scale)
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
                fusion.update(flip(sem_logits, -1))

        return fusion.output()


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes

    def _network(self, x, ret_intermediate=False):

        x_b = self.body(x)
        if isinstance(x_b, dict):
            x_b = x_b["out"]
        x_o = self.head(x_b)

        if ret_intermediate:
            return x_b, x_o
        return x_o

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def forward(self, x, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits = self.cls(out[1] if ret_intermediate else out)
        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits, {"body": out[0]}

        return sem_logits

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
