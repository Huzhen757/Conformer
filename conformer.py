import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

# Transformer Encoder中的MLP block
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features         # 第二个FC层中的节点个数
        hidden_features = hidden_features or in_features   # 第一个FC层中的节点个数
        self.fc1 = nn.Linear(in_features, hidden_features) # (384, 1536)
        self.act = act_layer()               
        self.fc2 = nn.Linear(hidden_features, out_features)# (1536, 384)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#  Multi-head Self-attention 
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 384/6=64 每个head中的token的维度
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5 # 计算Q和K的相似度时分母用到的数值=1/sqrt(64)=0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 一次FC同时得到Q，K以及V三个矩阵
        self.attn_drop = nn.Dropout(attn_drop) # dropout:0-0.2, 12个等差数列
        self.proj = nn.Linear(dim, dim) # 多个head的输出进行concat后，再做一次矩阵变换得到multi-head attention的结果
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, num_patches+1(class token), total_embed_dim]

        # qkv(x): [batch_size, num_patches+1, 3*total_embed_dim] = [batchsize, 197, 3*384]
        # reshape() -> permute: [batchsize, num_patches+1, 3, 6, 384/6] -> [3, batchsize, 6, num_patches+1, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 获取q，k以及v矩阵，[batchsize, 6, 197, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 将key矩阵的最后两个维度进行转置，高维矩阵乘法转换成两个维度的矩阵乘法 [batchsize, 6, 197, 64] * [batchsize, 6, 64, 197]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [batchsize, 6, 197, 197]
        attn = attn.softmax(dim=-1) # 在最后一个维度上进行softmax也就是针对每一行进行softmax
        attn = self.attn_drop(attn)
        # attention * v：[batchsize, 6, 197, 64] -> [batchsize, 197, 6, 64] -> [batchsize, 197, 384]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)      # 进行一个线性变换得到multi-head attention的输出 [batch, 197, 384]
        x = self.proj_drop(x)
        return x

# transformer分支上的block：Multihead-6 self-attention + MLP block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim) # layer norm 1
        self.attn = Attention(  
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) # layer norm 2
        mlp_hidden_dim = int(dim * mlp_ratio) # MLP block中的第一个FC层的hidden units数：384*4=1536
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# c2中的conv分支的第一个block：1x1conv -> 3x3conv -> 1x1conv 前两个conv的channel相同，最后一个1x1conv的channel是前面channel的4倍
class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion
        # 1x1 conv  (56, 56, 64) -> (56, 56, 64)
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)
        # 3x3 conv (56, 56, 64) -> (56, 56, 64)
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)
        # 1x1 conv 升维 (56, 56, 64) -> (56, 56, 256)
        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)
        # short cut (56, 56, 64) -> (56, 56, 256)
        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2: # 若该变量为True，表示需要将conv分支中的3x3卷积的输出进行转换到transformer分支中
            return x, x2
        else:
            return x  # 否则transformer的特张图经过转换与conv分支上的特张图fusion之后再进行conv block得到的输出


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        # 1x1 conv调整channel，avgpool调整分辨率
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W] -> [N, 384, H, W] 调整channel -> 384
        # maxpooling进行分辨率的下采样 [N,384,14,14] -> [N, 384, 196] -> [N, 196, 384]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)  
        x = self.act(x)
        # 取transformer输出的tensor的第二个维度上的第一个值即class_token上的值，再增加一个维度 [N,384]->[N,1,384]
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        # 再和conv分支山的特征图在维度1上进行concat -> [N, 197, 384]
        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()
        # Upsample + 1x1conv + batch norm
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()
    #transformer—>conv分支，获取除class token之外的所有的token进行操作
    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        # 使用双线性插值进行Up sampling得到conv分支上的特征图
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x

# 对应论文中的stage2-12的bottlneck
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        # 除去最后一个stage，stage2-11中的transformer分支转换成conv分支上之后，进行在conv分支上进行的卷积操作都没有short cut
        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)
        # conv分支经过1x1conv->Downsample->layer norm转换成transform分支上的特征图
        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
        # transformer分支经过Upsample -> 1x1conv -> batch norm转换成conv分支上的特征图
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x) # x作为conv分支上的特征图，x2作为transformer分支上的特征图

        _, _, H, W = x2.shape #获取conv分支中的特征图的h和w，用于进行下一步的down sampling 

        x_st = self.squeeze_block(x2, x_t) # conv分支上的特征图转换成transformer分支上

        x_t = self.trans_block(x_st + x_t) # feature fusion之后再进行multi-head attention

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)
        # 经过MHSA-6之后，transformer的特征图转换到conv分支上 [N, 197, 384] -> [N, 64, 56, 56]
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False) 
        # feature fusion之后在进行conv分支上的conv block(1x1conv->3x3conv->1x1conv)
        return x, x_t


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        # 在图像token的最前面加上一个class token(维度与图像token保持一致384)，原来是14*14个token，现在有14*14+1=197个token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, 384)
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        # 下面两个transformer分支上的class head:对embedding进行layernorm + 一个fc层(embed_dim, num_classes)进行分类
        self.trans_norm = nn.LayerNorm(embed_dim) # (384, )
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # (384, class)
        # 定义卷积分支上的class head: global average pooling + 一个fc层用于分类(1024, class)
        self.pooling = nn.AdaptiveAvgPool2d(1) # (1, 1, 1024)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes) # (1024, class)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        # 论文中的c1 block：conv+max pool (224, 224, 3) -> (112, 112, 64) -> (56, 56, 64)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio) # 256
        trans_dw_stride = patch_size // 4                   # 16 / 4 
        # C2中卷积分支的第一个block 
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1) 
        # C2中transformer分支的第一个block：使用4x4conv， (56, 56, 64) -> (16, 16, 384) 得到16x16个patches，维度384
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage 对应着C2中的后三个block
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )

# 对于C3中的stage5-8，只有第一次进入到C3时的卷积s=2(针对第一个bottleneck中的3x3conv而言的，下采样的过程) 以后三次的卷积s=1，
# 并且只有第一次的in_channel=256,以后的inchannel=512
        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1 
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False # 只有第一次进入到C3时才有short cut
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )
# 对于C4中的stage9-11，同上，只有第一次进入到C4中，只有第一个bottleneck中有short cut，其余的stage的bottleneck都没有short cut
# 注意：最后一个stage12，与前面的stage的操作都相反，即第一个bottleneck中不进行下采样，即3x3conv的s=1，并且也没有short cut，
# 而在stage12的第二个bottleneck中的3x3conv的s=2，进行下采样，并且存在short cut
        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage    
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion
                    )
            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x):
        B = x.shape[0]  # 获取batch_size
        cls_tokens = self.cls_token.expand(B, -1, -1) # class_token在第一个维度扩展batchsize倍
        # (1, 1, 384) -> (64, 1, 384)
        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] ->[N, 64, 112, 112] ->[N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage: 针对conv分支上的C2中的第一个bottleneck [N, 64, 56, 56] -> [N, 256, 56, 56]
        x = self.conv_1(x_base, return_x_2=False) 
        # 针对transformer分支上的C2中的第一个bottlenck [N, 64, 56, 56] -> [N, 384, 14, 14] -> [N, 384, 196] -> [N, 196, 384]
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1) # patches + class_token [N, 197, 384]
        x_t = self.trans_1(x_t) # transformer encoder [N, 197, 384]
        
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
        
        # conv classification [N, 1024, 7, 7] -> [N, 1, 1, 1024] -> [N, 1024]
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p) #  FC(1024, num_classes) [N, num_classes]

        # trans classification [N, 197, 384] -> layer norm -> [N, 384] -> [N, num_classes]
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0]) # FC(384, num_classes) 

        return [conv_cls, tran_cls]
