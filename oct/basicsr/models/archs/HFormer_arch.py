
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import warnings
from einops import rearrange
import pdb
# import pdb
import time
import matplotlib.pyplot as plt

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class A_MSA(nn.Module):
    def __init__(self,dim, num_heads,num_heads_v, bias):
        super(A_MSA, self).__init__()
        self.num_heads = num_heads_v
        self.temperature = nn.Parameter(torch.ones(num_heads_v, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q0,k0,v0 = qkv.chunk(3, dim=1)   
        q = rearrange(q0, 'b c  h (head w)  -> b  head w (c h)', head=self.num_heads)
        k = rearrange(k0, 'b c  h (head w)  -> b  head w (c h)', head=self.num_heads)
        v = rearrange(v0, 'b c  h (head w)  -> b  head w (c h)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head w (c h) -> b c h (head w)', head=self.num_heads, c=c, h=h)
        out = self.project_out(out)
        return out

class IA_MSA(nn.Module):
    def __init__(self,dim, num_heads,num_heads_v, bias):
        super(IA_MSA, self).__init__()
        self.num_heads = num_heads_v
        self.temperature = nn.Parameter(torch.ones(num_heads_v, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q0,k0,v0 = qkv.chunk(3, dim=1)
        q   = rearrange(q0, ' b c (head h) w  -> (b w) head h c', head=self.num_heads)
        k   = rearrange(k0, ' b c (head h) w  -> (b w) head h c', head=self.num_heads)
        v   = rearrange(v0, ' b c (head h) w  -> (b w) head h c', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, '(b w) head h c-> b c (head h) w', head=self.num_heads, b=b, w=w)
        out = self.project_out(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward_V(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_V, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class C_MSA(nn.Module):
    def __init__(self, dim, num_heads,num_heads_v, bias):
        super(C_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q0,k0,v0 = qkv.chunk(3, dim=1)   
        
        q = rearrange(q0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out




class TransformerBlock_V(nn.Module):
    def __init__(self, dim, num_heads, num_heads_v,ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_V, self).__init__()
        
        # C_MSA
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.c_msa = C_MSA(dim, num_heads,num_heads_v, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        # A_MSA
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.a_msa = A_MSA(dim, num_heads, num_heads_v,bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward_V(dim, ffn_expansion_factor, bias)

        # IA_MSA
        self.norm5 = LayerNorm(dim, LayerNorm_type)
        self.ia_msa = IA_MSA(dim, num_heads, num_heads_v,bias)
        self.norm6 = LayerNorm(dim, LayerNorm_type)
        self.ffn3 = FeedForward_V(dim, ffn_expansion_factor, bias)

        self.norm7 = LayerNorm(dim, LayerNorm_type)
        self.c_msa2 = C_MSA(dim, num_heads,num_heads_v, bias)
        self.norm8 = LayerNorm(dim, LayerNorm_type)
        self.ffn4 = FeedForward(dim, ffn_expansion_factor, bias)



    def forward(self, x):
        x = x + self.c_msa(self.norm1(x))
        x = x + self.ffn1(self.norm2(x))
        x = x + self.a_msa(self.norm3(x))
        x = x + self.ffn2(self.norm4(x))
        x = x + self.ia_msa(self.norm5(x))
        x = x + self.ffn3(self.norm6(x))
        x = x + self.c_msa2(self.norm7(x))
        x = x + self.ffn4(self.norm8(x))

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_heads_v,ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        
        # C_MSA
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.c_msa = C_MSA(dim, num_heads,num_heads_v, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.c_msa(self.norm1(x))
        x = x + self.ffn1(self.norm2(x))

        return x


class TransformerBlock_Refine(nn.Module):
    def __init__(self, dim, num_heads, num_heads_v, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_Refine, self).__init__()
        
        # C_MSA
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.c_msa = C_MSA(dim, num_heads, num_heads_v,bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.c_msa(self.norm1(x))
        x = x + self.ffn1(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##---------- Restormer -----------------------
class OSAT_HFormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        heads_v = [4,2,1,1],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   
        dual_pixel_task = False   
    ):

        super(OSAT_HFormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], num_heads_v=heads_v[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_v = TransformerBlock_V(dim=dim, num_heads=heads[0], num_heads_v=heads_v[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], num_heads_v=heads_v[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_v = TransformerBlock_V(dim=int(dim*2**1), num_heads=heads[1], num_heads_v=heads_v[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], num_heads_v=heads_v[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_v = TransformerBlock_V(dim=int(dim*2**2), num_heads=heads[2], num_heads_v=heads_v[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], num_heads_v=heads_v[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], num_heads_v=heads_v[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level3_v = TransformerBlock_V(dim=int(dim*2**2), num_heads=heads[2], num_heads_v=heads_v[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], num_heads_v=heads_v[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_v = TransformerBlock_V(dim=int(dim*2**1), num_heads=heads[1], num_heads_v=heads_v[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
 
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], num_heads_v=heads_v[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_v = TransformerBlock_V(dim=int(dim*2**1), num_heads=heads[0], num_heads_v=heads_v[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.refinement = nn.Sequential(*[TransformerBlock_Refine(dim=int(dim*2**1), num_heads=heads[0], num_heads_v=heads_v[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1_v(out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2_v(out_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3 = self.encoder_level3_v(out_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3_v(out_dec_level3)


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2_v(out_dec_level2) 


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1_v(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1
    



