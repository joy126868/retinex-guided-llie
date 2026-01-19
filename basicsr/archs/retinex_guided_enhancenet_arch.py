import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F


try:

    from .UNet_illu_arch import SG_UNet, SG_UNet_BilateralFilter_mask
except Exception:
    SG_UNet = None
    SG_UNet_BilateralFilter_mask = None

import warnings

from collections import OrderedDict

import torch.nn as nn

def GN(num_channels: int, num_groups: int = 32):
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


@ARCH_REGISTRY.register()
class NoisePriorHF(nn.Module):

    def __init__(self, blur_ks: int = 3, normalize: bool = True, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.blur_ks = int(blur_ks)
        self.normalize = bool(normalize)
        self.eps = float(eps)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"NoisePriorHF expects 4D tensor [B,C,H,W], got {x.shape}")
        if x.size(1) == 1:
            return x
        if x.size(1) >= 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b

        return x[:, 0:1]

    def forward(self, x: torch.Tensor):
        gray = self._to_gray(x)
        k = self.blur_ks
        blur = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)
        hf = (gray - blur).abs()
        if self.normalize:
            denom = hf.mean(dim=(2, 3), keepdim=True) + self.eps
            hf = hf / denom

        hf = torch.clamp(hf, 0.0, 10.0)
        return x, gray, hf


class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size//2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
            )

    def forward(self, input):
        return self.basic_unit(input)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class UNet_BilateralFilter_mask(nn.Module):
    def __init__(self, in_channels=4, channels=6, out_channels=1):
        super(UNet_BilateralFilter_mask,self).__init__()
        self.convpre = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv1 = UNetConvBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = UNetConvBlock(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = UNetConvBlock(4*channels, 4*channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = UNetConvBlock(8 * channels, 4 * channels)

        self.context2 = UNetConvBlock(2 * channels, 2 * channels)
        self.context1 = UNetConvBlock(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,4*channels,1,1,0),
                                    CALayer(4*channels,4),
                                    nn.Conv2d(4*channels,2*channels,3,1,1)
                                    )
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),
                                    CALayer(channels,2),
                                    nn.Conv2d(channels,channels,3,1,1)
                                    )

        self.conv_last = nn.Conv2d(channels,out_channels,3,1,1)


    def forward(self, x):
        x1 = self.conv1(self.convpre(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))

        x_global = self.Global(x3)
        _,_,h,w = x3.size()
        x_global = x_global.repeat(1,1,h,w)
        x3 = self.context_g(torch.cat([x_global,x3],1))

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))

        xout = self.conv_last(x1)

        return xout, x3


class UNetConvBlock_fre(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock_fre, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()



        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = UNetConvBlock_fre(self.split_len2, self.split_len1)
        self.G = UNetConvBlock_fre(self.split_len1, self.split_len2)
        self.H = UNetConvBlock_fre(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        out = torch.cat((y1, y2), 1)

        return out


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc,nc//2)

    def forward(self, x):
        return x+self.block(x)


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=3,padding=1,stride=1,groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,kernel_size=1,padding=0,stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFuse(nn.Module):
    def __init__(self, in_nc):
        super(SpatialFuse,self).__init__()

        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,3,1,1)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ChannelFuse(nn.Module):
    def __init__(self, in_nc):
        super(ChannelFuse,self).__init__()

        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,1,1,0)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.spa = SpatialFuse(nc)
        self.cha = ChannelFuse(nc)

    def forward(self,x):
        x = self.spa(x)
        x = self.cha(x)

        return x


class SelectiveSkipFusion(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, up, skip):
        x = torch.cat([up, skip], dim=1)
        g = self.gate(x)
        return torch.cat([up * (1.0 - g), skip * g], dim=1)


class ProcessNet(nn.Module):
    def __init__(self, nc):
        super(ProcessNet,self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)


        self.ssf1 = SelectiveSkipFusion(nc * 5)
        self.ssf2 = SelectiveSkipFusion(nc * 3)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        up3 = F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear')
        x34 = self.up1(self.ssf1(up3, x12))
        x4 = self.conv4(x34)
        up4 = F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear')
        x4 = self.up2(self.ssf2(up4, x01))
        x5 = self.conv5(x4)
        xout = self.convout(x5)

        return xout


class InteractNet(nn.Module):
    def __init__(self, inchannel, nc, outchannel):
        super(InteractNet,self).__init__()
        self.extract =  nn.Conv2d(inchannel, nc,1,1,0)
        self.process = ProcessNet(nc)
        self.recons = nn.Conv2d(nc, outchannel, 1, 1, 0)

    def forward(self, x):
        x_f = self.extract(x)
        x_f = self.process(x_f)+x_f
        y = self.recons(x_f)

        return y


class UNet_adjustment(nn.Module):
    def __init__(self, in_channels=4, channels=6, out_channels=1):
        super(UNet_adjustment,self).__init__()
        self.convpre = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv1 = UNetConvBlock_fre(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = UNetConvBlock_fre(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = UNetConvBlock_fre(4*channels, 4*channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = UNetConvBlock_fre(8 * channels, 4 * channels)

        self.context2 = UNetConvBlock_fre(2 * channels, 2 * channels)
        self.context1 = UNetConvBlock_fre(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,4*channels,1,1,0),
                                    CALayer(4*channels,4),
                                    nn.Conv2d(4*channels,2*channels,3,1,1)
                                    )
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),
                                    CALayer(channels,2),
                                    nn.Conv2d(channels,channels,3,1,1)
                                    )

        self.conv_last = nn.Conv2d(channels,out_channels,3,1,1)
        self.relu = nn.ReLU()


    def forward(self, x, ratio):
        x = torch.cat((x, ratio), 1)
        x1 = self.conv1(self.convpre(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))

        x_global = self.Global(x3)
        _,_,h,w = x3.size()
        x_global = x_global.repeat(1,1,h,w)
        x3 = self.context_g(torch.cat([x_global,x3],1))

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))

        xout = self.conv_last(x1)

        return self.relu(xout)


class IlluminationUpdateBlock(nn.Module):
    def __init__(self, illu_channel, mid_channels, kernel_size, unet_channel=None, prior_path=None):
        super(IlluminationUpdateBlock, self).__init__()

        self.prior_loaded = False









        pth = prior_path

        state = None
        if pth is not None and str(pth).strip() not in ["", "~", "None", "none", "null"]:
            try:
                ckpt = torch.load(pth, map_location="cpu")
                if isinstance(ckpt, dict):

                    if "params_ema" in ckpt and isinstance(ckpt["params_ema"], dict):
                        state = ckpt["params_ema"]
                    elif "params" in ckpt and isinstance(ckpt["params"], dict):
                        state = ckpt["params"]
                    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                        state = ckpt["state_dict"]
                    else:
                        state = ckpt
                else:
                    state = ckpt


                if isinstance(state, dict) and len(state) > 0:
                    keys = list(state.keys())
                    if all(k.startswith("module.") for k in keys):
                        state = {k[len("module."):]: v for k, v in state.items()}

                self.prior_loaded = isinstance(state, dict) and len(state) > 0

            except FileNotFoundError:
                warnings.warn(f"[IlluminationUpdateBlock] prior file not found: {pth}. Train from scratch.")
                state = None
                self.prior_loaded = False
            except Exception as e:
                warnings.warn(f"[IlluminationUpdateBlock] failed to load prior: {pth}. Train from scratch. ({e})")
                state = None
                self.prior_loaded = False


        use_sg = (SG_UNet_BilateralFilter_mask is not None)
        if isinstance(state, dict) and len(state) > 0:
            keys = list(state.keys())

            if any(k.startswith(("struct.", "enc", "bot.", "ssf", "dec", "adapt", "head")) for k in keys):
                use_sg = True

            elif any(k.startswith(("convpre.", "conv1.", "down1.", "conv2.", "down2.", "conv3.", "Global.", "context", "merge", "conv_last.")) for k in keys):
                use_sg = False


        if use_sg and (SG_UNet_BilateralFilter_mask is not None):
            self.L_learnedPrior = SG_UNet_BilateralFilter_mask(
                in_channels=1, out_channels=1, base_channels=32, return_feat=True
            )
            inferred_unet_channel = 32
        else:

            ch = 32
            if isinstance(state, dict) and "convpre.weight" in state:
                try:
                    ch = int(state["convpre.weight"].shape[0])
                except Exception:
                    ch = 32
            self.L_learnedPrior = UNet_BilateralFilter_mask(in_channels=1, channels=ch, out_channels=1)
            inferred_unet_channel = 4 * ch


        if unet_channel is None:
            unet_channel = inferred_unet_channel
        self._unet_channel = int(unet_channel)


        if isinstance(state, dict) and len(state) > 0:
            try:
                ret = self.L_learnedPrior.load_state_dict(state, strict=False)

                missing = getattr(ret, "missing_keys", [])
                unexpected = getattr(ret, "unexpected_keys", [])
                if len(missing) > 0 or len(unexpected) > 0:
                    warnings.warn(
                        f"[IlluminationUpdateBlock] prior loaded with mismatched keys. "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            except Exception as e:
                warnings.warn(f"[IlluminationUpdateBlock] load_state_dict failed (ignored): {e}")

        self.modulation_mul = nn.Sequential(
            nn.Conv2d(self._unet_channel, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, illu_channel, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.modulation_add = nn.Sequential(
            nn.Conv2d(self._unet_channel, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, illu_channel, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, low_light, illu, noise, refl, alpha, mu):

        out = self.L_learnedPrior(illu)
        if isinstance(out, (tuple, list)):
            L_prior, L_prior_feat = out
        else:
            L_prior, L_prior_feat = out, None

        L_cat = torch.cat([illu, illu, illu], 1)
        identity = torch.ones_like(L_cat)
        L_hat = (identity - alpha * refl * refl) * illu - alpha * refl * (noise - low_light)
        illu = torch.mean(L_hat, 1).unsqueeze(1)

        if L_prior_feat is not None:
            L_prior_feat = F.interpolate(
                L_prior_feat, size=illu.shape[-2:], mode='bilinear', align_corners=True
            )

            illu = illu + self.modulation_add(L_prior_feat)




        return illu, L_hat

class ReflectanceUpdateBlock(nn.Module):
    def __init__(self, refl_channel, mid_channels, kernel_size):
        super(ReflectanceUpdateBlock, self).__init__()
        self.prox = BasicUnit(refl_channel, mid_channels, refl_channel, kernel_size)

    def forward(self, low_light, illu, noise, refl, beta, mu):

        identity = torch.ones_like(illu)

        refl_hat = (identity - beta * illu * illu) * refl - beta * illu * (noise - low_light)
        refl = self.prox(refl_hat) + refl_hat


        return refl


class NoiseUpdateBlock(nn.Module):
    def __init__(self, noise_channel, mid_channels, kernel_size):
        super(NoiseUpdateBlock, self).__init__()
        self.prox = BasicUnit(noise_channel, mid_channels, noise_channel, kernel_size)

    def shrink(self, x, r):
        zeros = torch.zeros_like(x)
        z = torch.sign(x) * torch.max(torch.abs(x) - r, zeros)
        return z

    def forward(self, low_light, illu, refl, mu):
        illu_cat = torch.cat([illu, illu, illu], 1)
        noise_hat = self.shrink(low_light - refl * illu_cat, 1 / mu)
        noise = self.prox(noise_hat) + noise_hat


        return noise

def _sobel_grad(gray: torch.Tensor) -> torch.Tensor:

    if gray.dim() == 3:
        gray = gray.unsqueeze(1)
    if gray.size(1) != 1:
        gray = gray[:, :1, :, :]

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _soft_clamp01(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:

    return x - F.softplus(x - 1.0, beta=beta) + F.softplus(-x, beta=beta)
def _saturation_boost_keep_luma(x, s: float = 1.15):

    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    Y = (x * w).sum(dim=1, keepdim=True)
    out = Y + s * (x - Y)
    return torch.clamp(out, 0.0, 1.0)











import math

def _rgb_to_hvi(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    maxc, _ = torch.max(x, dim=1, keepdim=True)
    minc, _ = torch.min(x, dim=1, keepdim=True)
    v = maxc
    delt = maxc - minc
    s = delt / (maxc + eps)



    delt_safe = delt + eps
    rc = (maxc - r) / delt_safe
    gc = (maxc - g) / delt_safe
    bc = (maxc - b) / delt_safe

    h = torch.zeros_like(maxc)

    mask = (maxc == r).float()
    h = h + mask * (bc - gc)

    mask = (maxc == g).float()
    h = h + mask * (2.0 + rc - bc)

    mask = (maxc == b).float()
    h = h + mask * (4.0 + gc - rc)
    h = (h / 6.0) % 1.0

    h = torch.where(delt > eps, h, torch.zeros_like(h))

    ang = 2.0 * math.pi * h
    hx = s * torch.cos(ang)
    hy = s * torch.sin(ang)
    return torch.cat([hx, hy, v], dim=1)


def _hvi_to_rgb(hvi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    hx, hy, v = hvi[:, 0:1], hvi[:, 1:2], hvi[:, 2:3]
    s = torch.sqrt(hx * hx + hy * hy + eps)
    h = torch.atan2(hy, hx) / (2.0 * math.pi)
    h = (h % 1.0 + 1.0) % 1.0


    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    i = i.to(torch.int64) % 6

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)


    r = torch.zeros_like(v); g = torch.zeros_like(v); b = torch.zeros_like(v)
    cond = (i == 0); r = torch.where(cond, v, r); g = torch.where(cond, t, g); b = torch.where(cond, p, b)
    cond = (i == 1); r = torch.where(cond, q, r); g = torch.where(cond, v, g); b = torch.where(cond, p, b)
    cond = (i == 2); r = torch.where(cond, p, r); g = torch.where(cond, v, g); b = torch.where(cond, t, b)
    cond = (i == 3); r = torch.where(cond, p, r); g = torch.where(cond, q, g); b = torch.where(cond, v, b)
    cond = (i == 4); r = torch.where(cond, t, r); g = torch.where(cond, p, g); b = torch.where(cond, v, b)
    cond = (i == 5); r = torch.where(cond, v, r); g = torch.where(cond, p, g); b = torch.where(cond, q, b)

    return torch.cat([r, g, b], dim=1)


class HVIDecoupleRefiner(nn.Module):

    def __init__(self, base: int = 32, h_scale: float = 0.15, v_scale: float = 0.25):
        super().__init__()
        self.h_scale = float(h_scale)
        self.v_scale = float(v_scale)
        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.GELU(),
        )
        self.head_h = nn.Conv2d(base, 2, 3, 1, 1)
        self.head_v = nn.Conv2d(base, 1, 3, 1, 1)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        hvi = _rgb_to_hvi(torch.clamp(rgb01, 0.0, 1.0))
        feat = self.enc(hvi)
        dh = torch.tanh(self.head_h(feat)) * self.h_scale
        dv = torch.tanh(self.head_v(feat)) * self.v_scale

        hxhy = hvi[:, 0:2] + dh
        v = torch.clamp(hvi[:, 2:3] + dv, 0.0, 1.0)
        hvi2 = torch.cat([hxhy, v], dim=1)
        rgb2 = _hvi_to_rgb(hvi2)
        return torch.clamp(rgb2, 0.0, 1.0)


class HaarDWT(nn.Module):

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        self.register_buffer('k_ll', ll.view(1, 1, 2, 2))
        self.register_buffer('k_lh', lh.view(1, 1, 2, 2))
        self.register_buffer('k_hl', hl.view(1, 1, 2, 2))
        self.register_buffer('k_hh', hh.view(1, 1, 2, 2))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        k_ll = self.k_ll.repeat(c, 1, 1, 1)
        k_lh = self.k_lh.repeat(c, 1, 1, 1)
        k_hl = self.k_hl.repeat(c, 1, 1, 1)
        k_hh = self.k_hh.repeat(c, 1, 1, 1)
        ll = F.conv2d(x, k_ll, stride=2, padding=0, groups=c)
        lh = F.conv2d(x, k_lh, stride=2, padding=0, groups=c)
        hl = F.conv2d(x, k_hl, stride=2, padding=0, groups=c)
        hh = F.conv2d(x, k_hh, stride=2, padding=0, groups=c)
        return ll, lh, hl, hh


class HaarIDWT(nn.Module):

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        self.register_buffer('k_ll', ll.view(1, 1, 2, 2))
        self.register_buffer('k_lh', lh.view(1, 1, 2, 2))
        self.register_buffer('k_hl', hl.view(1, 1, 2, 2))
        self.register_buffer('k_hh', hh.view(1, 1, 2, 2))

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor):
        b, c, h, w = ll.shape
        k_ll = self.k_ll.repeat(c, 1, 1, 1)
        k_lh = self.k_lh.repeat(c, 1, 1, 1)
        k_hl = self.k_hl.repeat(c, 1, 1, 1)
        k_hh = self.k_hh.repeat(c, 1, 1, 1)
        x = F.conv_transpose2d(ll, k_ll, stride=2, padding=0, groups=c)
        x = x + F.conv_transpose2d(lh, k_lh, stride=2, padding=0, groups=c)
        x = x + F.conv_transpose2d(hl, k_hl, stride=2, padding=0, groups=c)
        x = x + F.conv_transpose2d(hh, k_hh, stride=2, padding=0, groups=c)
        return x


class WaveletPyramidHFRefiner(nn.Module):

    def __init__(self, base: int = 32, levels: int = 2, scale: float = 0.20):
        super().__init__()
        self.levels = int(levels)
        self.scale = float(scale)
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        def hf_net(in_ch: int):
            return nn.Sequential(
                nn.Conv2d(in_ch, base, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(base, base, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(base, in_ch - 1, 3, 1, 1),
            )


        self.hf_net1 = hf_net(10)
        self.hf_net2 = hf_net(10) if self.levels >= 2 else None


        self.gate_s = nn.Parameter(torch.tensor(6.0))
        self.gate_n = nn.Parameter(torch.tensor(8.0))
        self.gate_b = nn.Parameter(torch.tensor(0.0))

    def _gate(self, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:

        g = self.gate_s * s - self.gate_n * n + self.gate_b
        return torch.sigmoid(g)

    def forward(self, rgb01: torch.Tensor, structure: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        x0 = torch.clamp(rgb01, 0.0, 1.0)


        ll1, lh1, hl1, hh1 = self.dwt(x0)
        hf1 = torch.cat([lh1, hl1, hh1], dim=1)
        s1 = F.avg_pool2d(structure, 2, 2)
        n1 = F.avg_pool2d(noise_level, 2, 2)
        g1 = self._gate(s1, n1)
        d1 = torch.tanh(self.hf_net1(torch.cat([hf1, g1], dim=1)))
        hf1n = hf1 + self.scale * g1 * d1
        lh1n, hl1n, hh1n = torch.chunk(hf1n, 3, dim=1)


        ll1n = ll1
        if self.levels >= 2 and self.hf_net2 is not None:
            ll2, lh2, hl2, hh2 = self.dwt(ll1)
            hf2 = torch.cat([lh2, hl2, hh2], dim=1)
            s2 = F.avg_pool2d(structure, 4, 4)
            n2 = F.avg_pool2d(noise_level, 4, 4)
            g2 = self._gate(s2, n2)
            d2 = torch.tanh(self.hf_net2(torch.cat([hf2, g2], dim=1)))
            hf2n = hf2 + self.scale * g2 * d2
            lh2n, hl2n, hh2n = torch.chunk(hf2n, 3, dim=1)
            ll1n = self.idwt(ll2, lh2n, hl2n, hh2n)


        x1 = self.idwt(ll1n, lh1n, hl1n, hh1n)

        out = x0 + self.scale * (x1 - x0)
        return torch.clamp(out, 0.0, 1.0)


@ARCH_REGISTRY.register()





class _ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=True)
        self.norm = GN(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class _ResConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = _ConvAct(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.n2 = GN(ch)
    def forward(self, x):
        y = self.c1(x)
        y = self.n2(self.c2(y))
        return F.gelu(x + y)

class _TokenMHA(nn.Module):

    def __init__(self, dim, nhead=4, mlp_ratio=2.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.30))

    def forward(self, x, s=None):

        B, C, H, W = x.shape
        if s is not None:

            s_n = s / (s.mean(dim=(2, 3), keepdim=True) + 1e-6)
            x = x * (1.0 + self.alpha * torch.clamp(s_n, 0.0, 5.0))
        t = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        t2 = self.ln1(t)
        a, _ = self.attn(t2, t2, t2, need_weights=False)
        t = t + a
        t2 = self.ln2(t)
        t = t + self.mlp(t2)
        y = t.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y

class Stage2ResidualRefiner(nn.Module):

    def __init__(self, in_ch=8, base=48, nhead=4, use_ssf=True):
        super().__init__()
        self.in_proj = _ConvAct(in_ch, base, 3, 1, 1)

        self.e1 = _ResConv(base)
        self.d2 = _ConvAct(base, base*2, 4, 2, 1)
        self.e2 = _ResConv(base*2)
        self.d3 = _ConvAct(base*2, base*3, 4, 2, 1)
        self.e3 = _ResConv(base*3)
        self.d4 = _ConvAct(base*3, base*4, 4, 2, 1)
        self.e4 = _ResConv(base*4)
        self.d5 = _ConvAct(base*4, base*4, 4, 2, 1)
        self.b0 = _ResConv(base*4)
        self.t0 = _TokenMHA(base*4, nhead=nhead, mlp_ratio=2.0)


        self.u4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.f4 = _ConvAct(base*8, base*4, 3, 1, 1)
        self.u3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.f3 = _ConvAct(base*7, base*3, 3, 1, 1)
        self.u2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.f2 = _ConvAct(base*5, base*2, 3, 1, 1)
        self.u1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.f1 = _ConvAct(base*3, base, 3, 1, 1)

        self.use_ssf = bool(use_ssf)
        if self.use_ssf:

            self.ssf4 = SelectiveSkipFusion(in_ch=base*8)
            self.ssf3 = SelectiveSkipFusion(in_ch=base*7)
            self.ssf2 = SelectiveSkipFusion(in_ch=base*5)
            self.ssf1 = SelectiveSkipFusion(in_ch=base*3)

        self.out = nn.Conv2d(base, 3, 3, 1, 1)

        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def _align_to(self, x, ref):

        if x.shape[-2:] != ref.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=ref.shape[-2:], mode='bilinear', align_corners=False
            )
        return x

    def forward(self, x, structure=None):

        x0 = self.in_proj(x)
        e1 = self.e1(x0)
        e2 = self.e2(self.d2(e1))
        e3 = self.e3(self.d3(e2))
        e4 = self.e4(self.d4(e3))
        b = self.d5(e4)
        b = self.b0(b)

        s16 = None
        if structure is not None:
            s16 = F.avg_pool2d(structure, kernel_size=16, stride=16)


            if s16.shape[-2:] != b.shape[-2:]:
                s16 = torch.nn.functional.interpolate(
                    s16, size=b.shape[-2:], mode='bilinear', align_corners=False
                )

        b = self.t0(b, s=s16)


        d4 = self.u4(b)
        d4 = self._align_to(d4, e4)

        if self.use_ssf:
            cat4 = self.ssf4(d4, e4)
        else:
            cat4 = torch.cat([d4, e4], dim=1)

        d4 = self.f4(cat4)

        d3 = self.u3(d4)
        d3 = self._align_to(d3, e3)
        if self.use_ssf:
            cat3 = self.ssf3(d3, e3)
        else:
            cat3 = torch.cat([d3, e3], dim=1)
        d3 = self.f3(cat3)

        d2 = self.u2(d3)
        d2 = self._align_to(d2, e2)
        if self.use_ssf:
            cat2 = self.ssf2(d2, e2)
        else:
            cat2 = torch.cat([d2, e2], dim=1)
        d2 = self.f2(cat2)

        d1 = self.u1(d2)
        d1 = self._align_to(d1, e1)
        if self.use_ssf:
            cat1 = self.ssf1(d1, e1)
        else:
            cat1 = torch.cat([d1, e1], dim=1)
        d1 = self.f1(cat1)

        return self.out(d1)

@ARCH_REGISTRY.register()
class RetinexGuidedEnhanceNet(nn.Module):
    def __init__(self, stage, illu_channel, refl_channel, noise_channel, num_feat, ratio, alpha=0.001, beta=0.001, mu=0.1, illu_prior_path=None, freeze_illu_prior=True, **kwargs):
        super(RetinexGuidedEnhanceNet, self).__init__()

        self.illumination_update = IlluminationUpdateBlock(illu_channel, num_feat, 1, prior_path=illu_prior_path)

        if bool(freeze_illu_prior) and getattr(self.illumination_update, 'prior_loaded', False):
            for p in self.illumination_update.L_learnedPrior.parameters():
                p.requires_grad = False
        self.reflectance_update = ReflectanceUpdateBlock(3, num_feat, 1)
        self.noise_update = NoiseUpdateBlock(3, num_feat, 1)
        self.illumination_adjust_head = InteractNet(inchannel=4, nc=8, outchannel=1)


        init_illu_bias = float(kwargs.get('init_illu_bias', -1.2))
        try:
            if hasattr(self.illumination_adjust_head, 'recons') and getattr(self.illumination_adjust_head.recons, 'bias', None) is not None:
                nn.init.constant_(self.illumination_adjust_head.recons.bias, init_illu_bias)
        except Exception:
            pass
        self.reflectance_restore_head = (SG_UNet(in_channels=6, out_channels=3, base_channels=max(32, num_feat), residual=False)
                                  if SG_UNet is not None else InteractNet(inchannel=6, nc=8, outchannel=3))

        self.alpha = nn.Parameter(torch.tensor([alpha]), False)
        self.beta = nn.Parameter(torch.tensor([beta]), False)
        self.mu = nn.Parameter(torch.tensor([mu]))
        self.stage = stage
        self.ratio = ratio

        self.str_a = float(kwargs.get('str_a', 0.35))
        self.str_b = float(kwargs.get('str_b', 0.05))
        self.str_min = float(kwargs.get('str_min', 0.9))
        self.str_max = float(kwargs.get('str_max', 1.6))
        self.str_mix = float(kwargs.get('str_mix', 0.8))
        self.illu_bias = float(kwargs.get('illu_bias', 0.8))

        self.chroma_scale = float(kwargs.get('chroma_scale', 0.2))
        self.illu_gamma = float(kwargs.get('illu_gamma', 0.85))


        self.vis_highl_gamma = float(kwargs.get('vis_highl_gamma', 0.25))
        self.vis_highl_boost = float(kwargs.get('vis_highl_boost', 2.0))
        self.vis_color_only_enable = bool(kwargs.get('vis_color_only_enable', True))
        self.vis_saturation_scale = float(kwargs.get('vis_saturation_scale', 1.15))


        self.hvi_decouple_enable = bool(kwargs.get('hvi_decouple_enable', False))
        self.wavelet_refine_enable = bool(kwargs.get('wavelet_refine_enable', False))

        self.hvi_refiner = None
        if self.hvi_decouple_enable:
            self.hvi_refiner = HVIDecoupleRefiner(
                base=int(kwargs.get('hvi_refine_base', 32)),
                h_scale=float(kwargs.get('hvi_h_scale', 0.15)),
                v_scale=float(kwargs.get('hvi_v_scale', 0.25)),
            )

        self.wavelet_refiner = None
        if self.wavelet_refine_enable:
            self.wavelet_refiner = WaveletPyramidHFRefiner(
                base=int(kwargs.get('wavelet_base', 32)),
                levels=int(kwargs.get('wavelet_levels', 2)),
                scale=float(kwargs.get('wavelet_scale', 0.20)),
            )

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)


        self.edge_blur_ks = int(kwargs.get('edge_blur_ks', 5))
        self.edge_gate_thr = float(kwargs.get('edge_gate_thr', 0.15))
        self.edge_gate_tau = float(kwargs.get('edge_gate_tau', 0.05))
        self.edge_gate_base = float(kwargs.get('edge_gate_base', 0.10))
        self.edge_gate_pow = float(kwargs.get('edge_gate_pow', 2.0))

        self.noise_gate_thr = float(kwargs.get('noise_gate_thr', 0.25))
        self.noise_gate_tau = float(kwargs.get('noise_gate_tau', 0.08))
        self.noise_gate_pow = float(kwargs.get('noise_gate_pow', 1.5))
        self.delta_r_scale = float(kwargs.get('delta_r_scale', 0.5))

        self.refl_cap = float(kwargs.get('refl_cap', 5.0))

        self.delta_r_lf_ks = int(kwargs.get('delta_r_lf_ks', 15))
        self.delta_r_clip = float(kwargs.get('delta_r_clip', 0.25))

        self.use_soft_clamp = bool(kwargs.get('use_soft_clamp', True))
        self.softclamp_beta = float(kwargs.get('softclamp_beta', 10.0))


        self.wb_enable = bool(kwargs.get('wb_enable', True))
        self.wb_scale = float(kwargs.get('wb_scale', 0.15))
        self.wb_clip = float(kwargs.get('wb_clip', 0.25))
        self.wb_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )

        try:
            nn.init.zeros_(self.wb_mlp[-1].weight)
            nn.init.zeros_(self.wb_mlp[-1].bias)
        except Exception:
            pass
        self.wb_gain_last = None


        self.refiner_enable = bool(kwargs.get('refiner_enable', True))
        self.refiner_in_ch = int(kwargs.get('refiner_in_ch', 8))
        self.refiner_base = int(kwargs.get('refiner_base', 48))
        self.refiner_heads = int(kwargs.get('refiner_heads', 4))
        self.refiner_use_ssf = bool(kwargs.get('refiner_use_ssf', True))
        self.refiner_scale = float(kwargs.get('refiner_scale', 0.80))
        self.refiner = Stage2ResidualRefiner(
            in_ch=self.refiner_in_ch,
            base=self.refiner_base,
            nhead=self.refiner_heads,
            use_ssf=self.refiner_use_ssf
        ) if self.refiner_enable else None


    def unfolding(self, input_low_img):

        L_prior_cond = None
        illu = None
        refl = None
        noise = None
        for t in range(self.stage):
            if t == 0:
                illu = torch.max(input_low_img, 1)[0].unsqueeze(1)
                refl = input_low_img / (illu + 1e-8)
                noise = torch.zeros_like(input_low_img)
            else:
                illu, L_prior_cond = self.illumination_update(input_low_img, illu, noise, refl, self.alpha, self.mu)
                refl = self.reflectance_update(input_low_img, illu, noise, refl, self.beta, self.mu)
                noise = self.noise_update(input_low_img, illu, refl, self.mu)
        return refl, illu, noise, L_prior_cond

    def illumination_adjust(self, L, ratio, structure=None, noise_level=None):


        if torch.is_tensor(ratio):
            r = ratio
            if r.dim() == 1:
                r = r.view(-1, 1, 1, 1)
            elif r.dim() == 2:
                r = r.view(-1, 1, 1, 1)
            elif r.dim() == 4:
                pass
            else:
                r = r.view(-1, 1, 1, 1)
            ratio_map = r.expand(L.size(0), 1, L.size(2), L.size(3))
        else:
            ratio_map = torch.full_like(L, float(ratio), device=L.device)

        if structure is None:
            structure = torch.zeros_like(L)
        if noise_level is None:
            noise_level = torch.zeros_like(L)

        x_in = torch.cat([L, ratio_map, structure, noise_level], dim=1)
        High_L_raw = self.illumination_adjust_head(x_in)
        High_L_raw = torch.clamp(High_L_raw, -6.0, 6.0)


        meanL = torch.mean(L, dim=(2, 3), keepdim=True)
        strength = torch.clamp(self.str_a / (meanL + self.str_b), self.str_min, self.str_max)


        strength = 1.0 + self.str_mix * (strength - 1.0)
        bias = self.illu_bias
        High_L = 1.0 + (ratio_map - 1.0) * torch.sigmoid(High_L_raw * strength + bias)

        High_L = torch.clamp(High_L, min=1.0)
        High_L = torch.minimum(High_L, ratio_map)

        return High_L

    def forward(self, input_low_img):
        R, L, noise, L_prior_cond = self.unfolding(input_low_img)

        gray = 0.2989 * input_low_img[:, 0:1] + 0.5870 * input_low_img[:, 1:2] + 0.1140 * input_low_img[:, 2:3]
        k = int(self.edge_blur_ks)
        if k >= 3:
            gray = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        structure = torch.sqrt(gx * gx + gy * gy + 1e-12)
        structure = structure / (structure.mean(dim=(2, 3), keepdim=True) + 1e-6)
        noise_level = noise.abs().mean(dim=1, keepdim=True) if noise is not None else torch.zeros_like(L)

        gain_L = self.illumination_adjust(L, self.ratio, structure=structure,
                                          noise_level=noise_level)

        if (not self.training) or (torch.rand(1).item() < 0.001):
            with torch.no_grad():
                print("[gain_L] min/mean/max:",
                      gain_L.min().item(),
                      gain_L.mean().item(),
                      gain_L.max().item())


        refl_in = torch.cat([R, L, structure, noise_level], dim=1)

        s_guid = structure
        if s_guid is not None:
            denom = s_guid.mean(dim=(2, 3), keepdim=True) + 1e-6
            s_guid = torch.clamp(s_guid / denom, 0.0, 10.0)

        try:
            delta_R = self.reflectance_restore_head(refl_in, structure=s_guid)
        except TypeError:

            delta_R = self.reflectance_restore_head(refl_in)
        delta_R = torch.tanh(delta_R)


        lf_ks = int(getattr(self, 'delta_r_lf_ks', 15))
        if lf_ks >= 3:
            delta_R = delta_R - F.avg_pool2d(delta_R, kernel_size=lf_ks, stride=1, padding=lf_ks // 2)

        delta_clip = float(getattr(self, 'delta_r_clip', 0.25))
        if delta_clip > 0:
            delta_R = delta_clip * torch.tanh(delta_R / (delta_clip + 1e-6))



        s_edge = structure
        s_edge = s_edge / (s_edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        s_edge = torch.clamp(s_edge, 0.0, 1.0)
        gate_e = torch.sigmoid((s_edge - self.edge_gate_thr) / (self.edge_gate_tau + 1e-6))
        if self.edge_gate_pow != 1.0:
            gate_e = gate_e ** self.edge_gate_pow


        n = noise_level
        n = n / (n.mean(dim=(2, 3), keepdim=True) + 1e-6)
        n = torch.clamp(n, 0.0, 3.0) / 3.0
        gate_n = torch.sigmoid((n - self.noise_gate_thr) / (self.noise_gate_tau + 1e-6))
        if self.noise_gate_pow != 1.0:
            gate_n = gate_n ** self.noise_gate_pow


        gate = 1.0 - (1.0 - gate_e) * (1.0 - gate_n)
        gate = gate.detach()
        g = self.edge_gate_base + (1.0 - self.edge_gate_base) * gate


        delta_R = self.delta_r_scale * (delta_R * g)
        delta_Y = 0.2989 * delta_R[:, 0:1] + 0.5870 * delta_R[:, 1:2] + 0.1140 * delta_R[:, 2:3]
        delta_luma = delta_Y.repeat(1, 3, 1, 1)
        delta_chroma = delta_R - delta_luma
        delta_R = delta_luma + self.chroma_scale * delta_chroma

        cap = float(getattr(self, 'refl_cap', 5.0))
        R_safe = torch.clamp(R, 0.0, cap)
        restored_R = torch.clamp(R_safe + delta_R, 0.0, cap)



        if torch.is_tensor(self.ratio):
            r = self.ratio
            if r.dim() == 0:
                r = r.view(1, 1, 1, 1)
            elif r.dim() == 1:
                r = r.view(-1, 1, 1, 1)
            else:
                r = r.view(-1, 1, 1, 1)
            ratio_map = r.expand(L.size(0), 1, L.size(2), L.size(3))
        else:
            ratio_map = torch.full_like(L, float(self.ratio), device=L.device)

        L3 = L.repeat(1, 3, 1, 1)
        ratio_map3 = ratio_map.repeat(1, 3, 1, 1)
        High_L_illu = (L3 * gain_L).clamp_min(0.0)
        High_L_illu = torch.minimum(High_L_illu, ratio_map3)
        High_L_illu = High_L_illu.clamp_min(1e-6).pow(self.illu_gamma)


        if (not self.training) or (torch.rand(1).item() < 0.001):
            with torch.no_grad():
                print("[High_L_illu(L*gain)] min/mean/max:",
                      High_L_illu.min().item(),
                      High_L_illu.mean().item(),
                      High_L_illu.max().item())


        I_raw = High_L_illu * restored_R


        self.gain_L_last = gain_L.detach()

        denom = (ratio_map - 1.0).clamp_min(1e-6)
        High_L = torch.clamp((gain_L - 1.0) / denom, 0.0, 1.0)


        vis_g = float(getattr(self, "vis_highl_gamma", 0.25))
        High_L = High_L.clamp_min(1e-6).pow(vis_g)


        boost = float(getattr(self, "vis_highl_boost", 2.0))
        High_L = torch.clamp(High_L * boost, 0.0, 1.0)


        wb_gain = None
        if getattr(self, 'wb_enable', False):

            mean_rgb = input_low_img.mean(dim=(2, 3))
            mean_L = L.mean(dim=(2, 3))
            mean_n = noise_level.mean(dim=(2, 3))
            mean_s = structure.mean(dim=(2, 3))
            x = torch.cat([mean_rgb, mean_L, mean_n, mean_s], dim=1)
            logits = self.wb_mlp(x)
            gain = 1.0 + float(getattr(self, 'wb_scale', 0.15)) * torch.tanh(logits)



            w = gain.new_tensor([0.299, 0.587, 0.114]).view(1, 3)
            Y_before = (mean_rgb * w).sum(dim=1, keepdim=True)
            Y_after = ((mean_rgb * gain) * w).sum(dim=1, keepdim=True)
            k = (Y_before / (Y_after + 1e-6))
            gain = gain * k


            clipv = float(getattr(self, 'wb_clip', 0.25))
            if clipv > 0:
                gain = torch.clamp(gain, 1.0 - clipv, 1.0 + clipv)

            wb_gain = gain
            self.wb_gain_last = wb_gain.detach()
            I_raw = I_raw * wb_gain.view(-1, 3, 1, 1)


        if bool(getattr(self, 'use_soft_clamp', True)):
            I_stage0 = _soft_clamp01(I_raw, beta=float(getattr(self, 'softclamp_beta', 10.0)))
        else:
            I_stage0 = torch.clamp(I_raw, 0.0, 1.0)


        delta_ref = None
        I_final = I_stage0
        if getattr(self, 'refiner_enable', False) and (self.refiner is not None):
            ref_in = torch.cat([input_low_img, I_stage0, structure, noise_level], dim=1)
            delta_ref = self.refiner(ref_in, structure=structure)
            I_ref = I_stage0 + self.refiner_scale * delta_ref
            if bool(getattr(self, 'use_soft_clamp', True)):
                I_final = _soft_clamp01(I_ref, beta=float(getattr(self, 'softclamp_beta', 10.0)))
            else:
                I_final = torch.clamp(I_ref, 0.0, 1.0)




        if self.hvi_decouple_enable and (self.hvi_refiner is not None):
            I_final = self.hvi_refiner(I_final)





        if self.wavelet_refine_enable and (self.wavelet_refiner is not None):
            I_final = self.wavelet_refiner(I_final, structure=structure, noise_level=noise_level)

        if (not self.training) and getattr(self, "vis_color_only_enable", True):
            s = float(getattr(self, "vis_saturation_scale", 1.15))
            I_final = _saturation_boost_keep_luma(I_final, s=s)








        return I_final, High_L, L, restored_R, R, noise, L_prior_cond, I_raw, wb_gain, I_stage0, delta_ref
