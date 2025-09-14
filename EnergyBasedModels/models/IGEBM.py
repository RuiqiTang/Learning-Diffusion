import torch
import torch.nn as nn 
import torch.nn.functional as F

class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)
        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)
    
def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_class=None, downsample=False):
        super().__init__()

        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )
        )

        self.conv2 = spectral_norm(
            nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            ), std=1e-10, bound=True
        )

        self.class_embed = None

        if n_class is not None:
            # 修复：嵌入维度应为 out_channel * 4（每个卷积层需要weight和bias）
            class_embed = nn.Embedding(n_class, out_channel * 4)
            # 初始化：前半部分设为1（权重），后半部分设为0（偏置）
            class_embed.weight.data[:, :out_channel*2] = 1.0
            class_embed.weight.data[:, out_channel*2:] = 0.0
            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
            )

        self.downsample = downsample

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None and class_id is not None:
            # 获取类嵌入并调整形状
            embed = self.class_embed(class_id.long()).view(input.shape[0], -1, 1, 1)
            # 将嵌入分割为4部分：两个卷积层的权重和偏置
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            
            # 确保权重与输出的通道数匹配
            # 这里添加了维度检查，便于调试
            assert weight1.size(1) == out.size(1), \
                f"weight1通道数({weight1.size(1)})与out通道数({out.size(1)})不匹配"
            
            out = weight1 * out + bias1

        out = F.leaky_relu(out, negative_slope=0.2)

        out = self.conv2(out)

        if self.class_embed is not None and class_id is not None:
            assert weight2.size(1) == out.size(1), \
                f"weight2通道数({weight2.size(1)})与out通道数({out.size(1)})不匹配"
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        # 确保跳跃连接与输出的维度匹配
        if skip.size() != out.size():
            # 如果尺寸不匹配，使用插值调整
            skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            
        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = F.leaky_relu(out, negative_slope=0.2)

        return out

class IGEBM(nn.Module):
    def __init__(self, n_class=None):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv2d(1, 128, 3, padding=1), std=1)

        self.blocks = nn.ModuleList(
            [
                ResBlock(128, 128, n_class, downsample=True),
                ResBlock(128, 128, n_class),
                ResBlock(128, 256, n_class, downsample=True),
                ResBlock(256, 256, n_class),
                ResBlock(256, 256, n_class, downsample=True),
                ResBlock(256, 256, n_class),
            ]
        )

        self.linear = nn.Linear(256, 1)

    def forward(self, input, class_id=None):
        out = self.conv1(input)
        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out
