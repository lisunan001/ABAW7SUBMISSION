


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from utils.helpers import maybe_download
from .modules import *


from torch.nn import LayerNorm
from torch.nn import MultiheadAttention
from torch.nn.modules import ModuleList
import copy

data_info = {
    7: 'VOC',
}

models_urls = {
    '101_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/Owmttk9bdPROwc6/download',
    '18_imagenet' : 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '50_imagenet' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

bottleneck_idx = 0
save_idx = 0


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias)

def convfc(in_planes, out_planes, stride=1, bias=False):
    "for change to fc"
    return nn.Conv2d(in_planes, out_planes, kernel_size=7,
                                    stride=stride, padding=0, bias=bias)


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_stages, num_parallel):
        super(CRPBlock, self).__init__()
        for i in range(num_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes, out_planes))
        self.stride = 1
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.num_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = x + top
        return x


stages_suffixes = {0 : '_conv', 1 : '_conv_relu_varout_dimred'}

class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, num_stages, num_parallel):
        super(RCUBlock, self).__init__()
        for i in range(num_blocks):
            for j in range(num_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, bias=(j == 0)))
        self.stride = 1
        self.num_blocks = num_blocks
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        for i in range(self.num_blocks):
            residual = x
            for j in range(self.num_stages):
                x = self.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x = x + residual
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)#BatchNorm2dParallel(planes, num_parallel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)#BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
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

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)#BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)#BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes)#BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = nn.ReLU(inplace=True)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #if len(x) > 1:
        #    out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class TransformerEncoder(nn.Module):


    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):

        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):


    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2,weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class RefineNet(nn.Module):
    def __init__(self, block, layers, num_parallel, num_classes=7, bn_threshold=2e-2):
        self.inplanes = 64
        self.step = 15
        self.num_parallel = num_parallel
        super(RefineNet, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)#BatchNorm2dParallel(64, num_parallel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)



        #self.clf_conv = conv3x3(256, num_classes, bias=True)
        self.clf_fc_pre = convfc(512, 512, bias=True)#change for fc

        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.alpha = nn.Parameter(torch.ones([1, num_parallel, 157, 157], requires_grad=True))
        self.register_parameter('alpha', self.alpha)

        self.fc1 = nn.Linear(512*self.step , 256)

        self.fc2 = nn.Linear(256, num_classes)
        self.fc2_v = nn.Linear(256, 2)
        self.fc2_a = nn.Linear(256, 2)

    def _make_crp(self, in_planes, out_planes, num_stages):
        layers = [CRPBlock(in_planes, out_planes, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, num_blocks, num_stages):
        layers = [RCUBlock(in_planes, out_planes, num_blocks, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)  #BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)



    def forward(self, x):

        #self.rnn.flatten_parameters()
        batch, steps, C, H, W = x.size()

        x_in_0= x.view(batch * steps, C, H, W)

        x_in=x_in_0
        x_in = self.conv1(x_in)
        x_in = self.bn1(x_in)
        x_in = self.relu(x_in)
        x_in = self.maxpool(x_in)

        l1 = self.layer1(x_in)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout(l4)




        #out = self.clf_conv(x1)
        out_pre = self.clf_fc_pre(l4)#change for fc
        r_in_1=out_pre.view(batch, steps, -1)

        #r_in_3 = out_pre[2].view(batch, steps, -1)(1,0,2)
        r_in_1_t=r_in_1.permute(0,2,1)





        dsb_in_1=r_in_1_t#dsb_1.permute(0,2,1)




        x_1 = F.relu(self.fc1(dsb_in_1[:, :, :].reshape(batch,512*self.step)))

        x_v = self.fc2_v(x_1)
        x_a = self.fc2_a(x_1)
        #print(x_3.shape)
        #print('sb')
        x_1 = self.fc2(x_1)



        #x_3 = self.fc2(x_3)
        alpha_soft = F.softmax(self.alpha)
        out=[x_1]



        return out, alpha_soft,x_v,x_a


def refinenet(num_layers, num_classes, num_parallel, bn_threshold):
    if int(num_layers) == 50:
        layers = [3, 4, 6, 3]
    if int(num_layers) == 18:
        layers = [2, 2, 2, 2]
    elif int(num_layers) == 101:
        layers = [3, 4, 23, 3]
    elif int(num_layers) == 152:
        layers = [3, 8, 36, 3]
    else:
        print('invalid num_layers')
    #BasicBlockBottleneck
    model = RefineNet(BasicBlock, layers, num_parallel, num_classes, bn_threshold)
    return model


def model_init(model, num_layers, num_parallel, imagenet=False, pretrained=True):
    if imagenet:
        key = str(num_layers) + '_imagenet'
        url = models_urls[key]
        state_dict = maybe_download(key, url)
        model_dict = expand_model_dict(model.state_dict(), state_dict, num_parallel)
        model.load_state_dict(model_dict, strict=True)
    elif pretrained:
        dataset = data_info.get(7, None)
        if dataset:
            bname = str(num_layers) + '_' + dataset.lower()
            key = 'rf' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict
