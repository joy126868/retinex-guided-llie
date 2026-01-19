
from basicsr.losses.losses import L_color
import os
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from PIL import Image, ImageDraw, ImageFont

def _to_pil_rgb_from_bgr_np(img):

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255.0).astype(np.uint8)


    if arr.shape[2] == 3:
        arr = arr[..., ::-1]

    return Image.fromarray(arr).convert("RGB")


def make_grid_2x4_with_titles_center(img_list, titles, pad=10, title_h=50, font_size=24):

    assert len(img_list) == 8 and len(titles) == 8

    pil_imgs = [_to_pil_rgb_from_bgr_np(im) for im in img_list]
    ref = next((im for im in pil_imgs if im is not None), None)
    if ref is None:
        return None

    W, H = ref.size
    for i in range(8):
        if pil_imgs[i] is None:
            pil_imgs[i] = Image.new("RGB", (W, H), (255, 255, 255))
        elif pil_imgs[i].size != (W, H):
            pil_imgs[i] = pil_imgs[i].resize((W, H), Image.BILINEAR)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    tile_w, tile_h = W, H + title_h
    grid_w = 4 * tile_w + 5 * pad
    grid_h = 2 * tile_h + 3 * pad
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx in range(8):
        r, c = idx // 4, idx % 4
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)


        draw.rectangle([x0, y0, x0 + tile_w, y0 + title_h], fill=(245, 245, 245))

        t = titles[idx]

        try:
            bbox = draw.textbbox((0, 0), t, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(t, font=font)

        tx = x0 + (tile_w - tw) // 2
        ty = y0 + (title_h - th) // 2
        draw.text((tx, ty), t, fill=(0, 0, 0), font=font)


        canvas.paste(pil_imgs[idx], (x0, y0 + title_h))

    return canvas


def norm01_uint8(x, eps=1e-6):

    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    x = (x - mn) / (mx - mn + eps)
    x = np.clip(x, 0, 1)
    return (x * 255.0).astype(np.uint8)





def _to_3ch(img):

    if img is None:
        return None
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img

def _hstack(imgs, gap=10, gap_val=255):
    imgs = [_to_3ch(im) for im in imgs if im is not None]
    if len(imgs) == 0:
        return None
    h = imgs[0].shape[0]
    gap_arr = np.full((h, gap, 3), gap_val, dtype=imgs[0].dtype)
    out = imgs[0]
    for im in imgs[1:]:

        if im.shape[0] != h:
            continue
        out = np.concatenate([out, gap_arr, im], axis=1)
    return out


def _to_gray(x):

    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

def _sobel_grad(x):

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

def _norm01(x, eps=1e-6):

    x_min = x.amin(dim=(2,3), keepdim=True)
    x_max = x.amax(dim=(2,3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _vis01(x, k=2.5, eps=1e-6):

    mu = x.mean(dim=(2,3), keepdim=True)
    sig = x.std(dim=(2,3), keepdim=True).clamp_min(eps)
    lo = mu - k * sig
    hi = mu + k * sig
    y = (x - lo) / (hi - lo + eps)
    return y.clamp(0.0, 1.0)


def _soft_clamp01(x, beta=10.0):

    return x - F.softplus(x - 1.0, beta=beta) + F.softplus(-x, beta=beta)

def _gtmean_weight(mu_y: torch.Tensor, mu_fx: torch.Tensor, sigma: float = 0.1, eps: float = 1e-12) -> torch.Tensor:

    s = float(sigma)
    sig_y = s * mu_y
    sig_fx = s * mu_fx
    var_y = sig_y * sig_y + eps
    var_fx = sig_fx * sig_fx + eps
    var_sum = var_y + var_fx
    term1 = 0.25 * (mu_y - mu_fx).pow(2) / (var_sum + eps)
    term2 = 0.5 * torch.log((var_sum + eps) / (2.0 * sig_y * sig_fx + eps) + eps)
    db = term1 + term2
    return torch.clamp(db, 0.0, 1.0)

def _rgb_to_hv(x, eps=1e-6):

    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delt = maxc - minc
    s = delt / (maxc + eps)

    h = torch.zeros_like(maxc)
    mask = delt > eps


    is_r = (maxc == r) & mask
    is_g = (maxc == g) & mask
    is_b = (maxc == b) & mask

    h[is_r] = ((g - b)[is_r] / delt[is_r]) % 6.0
    h[is_g] = ((b - r)[is_g] / delt[is_g]) + 2.0
    h[is_b] = ((r - g)[is_b] / delt[is_b]) + 4.0
    h = (h / 6.0) % 1.0

    ang = 2.0 * torch.pi * h
    hv1 = torch.cos(ang) * s
    hv2 = torch.sin(ang) * s
    return torch.stack([hv1, hv2], dim=1)


def _rgb_to_hvi(x, eps=1e-6):

    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delt = maxc - minc
    s = delt / (maxc + eps)

    h = torch.zeros_like(maxc)
    mask = delt > eps
    is_r = (maxc == r) & mask
    is_g = (maxc == g) & mask
    is_b = (maxc == b) & mask
    h[is_r] = ((g - b)[is_r] / (delt[is_r] + eps)) % 6.0
    h[is_g] = ((b - r)[is_g] / (delt[is_g] + eps)) + 2.0
    h[is_b] = ((r - g)[is_b] / (delt[is_b] + eps)) + 4.0
    h = (h / 6.0) % 1.0

    ang = 2.0 * torch.pi * h
    hx = torch.cos(ang) * s
    hy = torch.sin(ang) * s
    return torch.stack([hx, hy, v], dim=1)


def _haar_dwt(x: torch.Tensor):

    b, c, h, w = x.shape

    if (h % 2) == 1:
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        h += 1
    if (w % 2) == 1:
        x = F.pad(x, (0, 1, 0, 0), mode='replicate')
        w += 1
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _wavelet_hf_loss(a: torch.Tensor, b: torch.Tensor, levels: int = 2):

    la, lb = a, b
    loss = 0.0
    for _ in range(int(levels)):
        ll_a, lh_a, hl_a, hh_a = _haar_dwt(la)
        ll_b, lh_b, hl_b, hh_b = _haar_dwt(lb)

        loss = loss + (lh_a - lh_b).abs().mean()
        loss = loss + (hl_a - hl_b).abs().mean()
        loss = loss + (hh_a - hh_b).abs().mean()
        la, lb = ll_a, ll_b
    return loss

def _blur(x: torch.Tensor, ks: int = 31) -> torch.Tensor:
    ks = int(ks)
    if ks < 3:
        return x
    if ks % 2 == 0:
        ks += 1
    pad = ks // 2
    c = x.size(1)
    weight = torch.ones(c, 1, ks, ks, device=x.device, dtype=x.dtype) / float(ks * ks)
    return torch.nn.functional.conv2d(x, weight, padding=pad, groups=c)


def _chroma2(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    g = x[:, 1:2].clamp(min=eps)
    c1 = torch.log(x[:, 0:1].clamp(min=eps) / g)
    c2 = torch.log(x[:, 2:3].clamp(min=eps) / g)
    return torch.cat([c1, c2], dim=1)

def _rgb_to_ycbcr(x: torch.Tensor):

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


def _np_uint8_bgr_to_torch_rgb01(img_bgr_uint8):

    if img_bgr_uint8 is None:
        raise ValueError("img is None")
    x = img_bgr_uint8.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[2] >= 3:
        x = x[:, :, :3]
    x = x[:, :, ::-1].copy()
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return t


def _torch01_to_lpips_input(x01):

    return x01 * 2.0 - 1.0

@MODEL_REGISTRY.register()
class RetinexGuidedLLIEModel(BaseModel):


    def __init__(self, opt):
        super(RetinexGuidedLLIEModel, self).__init__(opt)


        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.net_noisePrior = build_network(opt['network_noisePrior'])
        self.net_noisePrior = self.model_to_device(self.net_noisePrior)


        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)


        load_path_noisePrior = self.opt['path'].get('pretrain_network_noisePrior', None)
        if load_path_noisePrior is not None:
            param_key = self.opt['path'].get('param_key_decom', 'params')
            self.load_network(self.net_noisePrior, load_path_noisePrior, self.opt['path'].get('strict_load_noisePrior', True), param_key)

        if self.is_train:
            self.init_training_settings()

        if hasattr(self, 'net_noisePrior') and (self.net_noisePrior is not None):
            for p in self.net_noisePrior.parameters():
                p.requires_grad = False
            self.net_noisePrior.eval()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']


        self.sat_w = train_opt.get('sat_w', 0.0)
        self.mean_w = train_opt.get('mean_w', 0.0)
        self.lr_pix_w = train_opt.get('lr_pix_w', 0.0)
        self.lr_pix_scale = train_opt.get('lr_pix_scale', 0.25)

        self.use_gtmean_pix = bool(train_opt.get('use_gtmean_pix', False))
        self.gtmean_sigma = float(train_opt.get('gtmean_sigma', 0.10))


        self.use_gtmean_all = bool(train_opt.get('use_gtmean_all', False))
        self.use_gtmean_edge = bool(train_opt.get('use_gtmean_edge', self.use_gtmean_all))
        self.use_gtmean_hvi = bool(train_opt.get('use_gtmean_hvi', self.use_gtmean_all))
        self.use_gtmean_wavelet = bool(train_opt.get('use_gtmean_wavelet', self.use_gtmean_all))


        self.wavelet_w = float(train_opt.get('wavelet_w', 0.0))
        self.wavelet_levels = int(train_opt.get('wavelet_levels', 2))


        self.wb_reg_w = float(train_opt.get('wb_reg_w', 0.0))


        self.chroma_w = float(train_opt.get('chroma_w', 0.0))
        self.chroma_warmup_iters = int(train_opt.get('chroma_warmup_iters', 0))


        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')



            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('gtRecon_opt'):
            self.cri_gtRecon = build_loss(train_opt['gtRecon_opt']).to(self.device)
        else:
            self.cri_gtRecon = None

        if train_opt.get('lowRecon_opt'):
            self.cri_lowRecon = build_loss(train_opt['lowRecon_opt']).to(self.device)
        else:
            self.cri_lowRecon = None

        if train_opt.get('refl_opt'):
            self.cri_refl = build_loss(train_opt['refl_opt']).to(self.device)
        else:
            self.cri_refl = None

        if train_opt.get('illuMutualInput_opt'):
            self.cri_illuMutualInput = build_loss(train_opt['illuMutualInput_opt']).to(self.device)
        else:
            self.cri_illuMutualInput = None

        if train_opt.get('illuMutual_opt'):
            self.cri_illuMutual = build_loss(train_opt['illuMutual_opt']).to(self.device)
        else:
            self.cri_illuMutual = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('enhancedIllu_opt'):
            self.cri_enhancedIllu = build_loss(train_opt['enhancedIllu_opt']).to(self.device)
        else:
            self.cri_enhancedIllu = None

        self.enhancedIlluTV_w = 0.0
        if train_opt.get('enhancedIlluTV_opt'):
            self.enhancedIlluTV_w = float(train_opt['enhancedIlluTV_opt'].get('loss_weight', 0.0) or 0.0)
        self.enhancedIlluTV_warmup_iters = int(train_opt.get('enhancedIlluTV_warmup_iters', 0) or 0)

        if train_opt.get('reflRestore_opt'):
            self.cri_reflRestore = build_loss(train_opt['reflRestore_opt']).to(self.device)
        else:
            self.cri_reflRestore = None

        if train_opt.get('noisePrior_opt'):
            self.cri_noisePrior = build_loss(train_opt['noisePrior_opt']).to(self.device)
        else:
            self.cri_noisePrior = None


        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        if train_opt.get('hvi_opt'):
            self.cri_hvi = build_loss(train_opt['hvi_opt']).to(self.device)
        else:
            self.cri_hvi = None


        self.refl_warmup_iters = int(train_opt.get('refl_warmup_iters', 0) or 0)
        self.edge_warmup_iters = int(train_opt.get('edge_warmup_iters', 0) or 0)
        self.hvi_warmup_iters  = int(train_opt.get('hvi_warmup_iters', 0) or 0)


        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']


        self.sat_w = train_opt.get('sat_w', 0.0)
        self.mean_w = train_opt.get('mean_w', 0.0)
        self.lr_pix_w = train_opt.get('lr_pix_w', 0.0)
        self.lr_pix_scale = train_opt.get('lr_pix_scale', 0.25)

        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        loss_dict = OrderedDict()


        out_net = self.net_g(self.lq)




        self.output, self.enhanced_L, self.L, self.restored_R, self.R, self.noise, self.L_hat = out_net[:7]


        output_raw = out_net[7] if len(out_net) > 7 else self.output
        self.wb_gain = out_net[8] if len(out_net) > 8 else None

        self.stage0 = out_net[9] if len(out_net) > 9 else None
        self.delta_ref = out_net[10] if len(out_net) > 10 else None



        output_for_loss = self.output


        if bool(self.opt['train'].get('loss_soft_clamp', False)):
            output_for_loss = _soft_clamp01(output_for_loss)


        self.output_raw = output_raw


        self.output = torch.clamp(output_for_loss, 0.0, 1.0)


        _gtmean_W = None
        _gtmean_scale = None
        _gtmean_wmean = None
        if (self.use_gtmean_all or self.use_gtmean_pix or self.use_gtmean_edge or self.use_gtmean_hvi or self.use_gtmean_wavelet):
            mu_y = self.gt.mean(dim=(1, 2, 3))
            mu_fx = output_for_loss.mean(dim=(1, 2, 3))
            _gtmean_W = _gtmean_weight(mu_y, mu_fx, sigma=float(self.gtmean_sigma)).detach()
            _gtmean_scale = (mu_y / (mu_fx + 1e-6)).view(-1, 1, 1, 1)
            _gtmean_wmean = _gtmean_W.mean()
            loss_dict['gtmean_W'] = _gtmean_wmean



        refl_cap = float(self.opt['train'].get('refl_cap', getattr(self.net_g, 'refl_cap', 5.0)))
        self.refl_cap = refl_cap






        self.L = torch.clamp(self.L, 0.0, 1.0)

        self.R = torch.clamp(self.R, 0.0, refl_cap)
        self.restored_R = torch.clamp(self.restored_R, 0.0, refl_cap)


        with torch.no_grad():
            gt_gray = _to_gray(self.gt)

            gt_L = torch.nn.functional.avg_pool2d(gt_gray, kernel_size=15, stride=1, padding=7)
            gt_L = torch.clamp(gt_L, 1e-3, 1.0)
            gt_R = self.gt / gt_L
            gt_R = torch.clamp(gt_R, 0.0, refl_cap)
            self.gt_L = gt_L
            self.gt_R = gt_R
            self.gt_noise = torch.zeros_like(gt_L)

        _, _, self.output_noisePrior = self.net_noisePrior(output_for_loss)
        with torch.no_grad():
            _, _, self.gt_noisePrior = self.net_noisePrior(self.gt)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_lowRecon:
            l_lowRecon = self.cri_lowRecon(self.L * self.R + self.noise, self.lq)
            l_total += l_lowRecon
            loss_dict['l_lowRecon'] = l_lowRecon


        if self.cri_refl:

            l_refl = self.cri_refl(self.R, self.gt_R)
            l_total += l_refl
            loss_dict['l_refl'] = l_refl

        if self.cri_illuMutualInput:
            l_illuMutualInputLQ = self.cri_illuMutualInput(self.L, self.lq)
            l_total += l_illuMutualInputLQ
            loss_dict['l_illuMutualInputLQ'] = l_illuMutualInputLQ

            l_illuMutualInputGT = self.cri_illuMutualInput(self.gt_L, self.gt)
            l_total += l_illuMutualInputGT
            loss_dict['l_illuMutualInputGT'] = l_illuMutualInputGT

        if self.cri_illuMutual:
            l_illuMutual = self.cri_illuMutual(self.L, self.gt_L)
            l_total += l_illuMutual
            loss_dict['l_illuMutual'] = l_illuMutual

        if self.cri_pix:

            if getattr(self, 'use_gtmean_pix', False) and (_gtmean_scale is not None) and (_gtmean_wmean is not None):
                out_align = output_for_loss * _gtmean_scale
                l_pix_ori = self.cri_pix(output_for_loss, self.gt)
                l_pix_align = self.cri_pix(out_align, self.gt)
                l_pix = _gtmean_wmean * l_pix_ori + (1.0 - _gtmean_wmean) * l_pix_align
                loss_dict['l_pix_ori'] = l_pix_ori
                loss_dict['l_pix_align'] = l_pix_align
            else:
                l_pix = self.cri_pix(output_for_loss, self.gt)

            l_total += l_pix
            loss_dict['l_pix'] = l_pix


        hf_w = float(self.opt['train'].get('hf_w', 0.0))
        if hf_w > 0:
            def _lap(x):

                k = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], device=x.device, dtype=x.dtype).view(1,1,3,3)

                return torch.cat([F.conv2d(x[:,i:i+1], k, padding=1) for i in range(3)], dim=1)

            def _ms_lap_loss(a, b, levels=2):
                la, lb = a, b
                loss = 0.0
                for _ in range(levels):
                    loss = loss + (_lap(la) - _lap(lb)).abs().mean()
                    la = F.avg_pool2d(la, 2, 2)
                    lb = F.avg_pool2d(lb, 2, 2)
                return loss / float(levels)

            l_hf = _ms_lap_loss(output_for_loss, self.gt, levels=int(self.opt['train'].get('hf_levels', 2)))
            l_total += hf_w * l_hf
            loss_dict['l_hf'] = l_hf


        if getattr(self, 'wavelet_w', 0.0) > 0:
            if getattr(self, 'use_gtmean_wavelet', False) and (_gtmean_scale is not None) and (_gtmean_wmean is not None):
                l_wav_ori = _wavelet_hf_loss(output_for_loss, self.gt, levels=int(getattr(self, 'wavelet_levels', 2)))
                l_wav_align = _wavelet_hf_loss(output_for_loss * _gtmean_scale, self.gt, levels=int(getattr(self, 'wavelet_levels', 2)))
                l_wav = _gtmean_wmean * l_wav_ori + (1.0 - _gtmean_wmean) * l_wav_align
                loss_dict['l_wav_ori'] = l_wav_ori
                loss_dict['l_wav_align'] = l_wav_align
            else:
                l_wav = _wavelet_hf_loss(output_for_loss, self.gt, levels=int(getattr(self, 'wavelet_levels', 2)))

            l_total += float(getattr(self, 'wavelet_w', 0.0)) * l_wav
            loss_dict['l_wav'] = l_wav


        stage0_w = float(self.opt['train'].get('stage0_w', 0.0))
        if stage0_w > 0 and getattr(self, 'stage0', None) is not None:
            l_stage0 = self.cri_pix(self.stage0, self.gt) if self.cri_pix else (self.stage0 - self.gt).abs().mean()
            l_total += stage0_w * l_stage0
            loss_dict['l_stage0'] = l_stage0


        delta_tv_w = float(self.opt['train'].get('delta_tv_w', 0.0))
        if delta_tv_w > 0 and getattr(self, 'delta_ref', None) is not None:
            dx = (self.delta_ref[:, :, :, 1:] - self.delta_ref[:, :, :, :-1]).abs().mean()
            dy = (self.delta_ref[:, :, 1:, :] - self.delta_ref[:, :, :-1, :]).abs().mean()
            l_tv = dx + dy
            l_total += delta_tv_w * l_tv
            loss_dict['l_delta_tv'] = l_tv




        if getattr(self, 'lr_pix_w', 0.0) and self.lr_pix_w > 0:
            scale = float(getattr(self, 'lr_pix_scale', 0.25))
            if 0.0 < scale < 1.0:
                out_lr = F.interpolate(output_for_loss, scale_factor=scale, mode='area')
                gt_lr = F.interpolate(self.gt, scale_factor=scale, mode='area')
                l_lr = torch.mean(torch.abs(out_lr - gt_lr))
                l_total += self.lr_pix_w * l_lr
                loss_dict['l_lr_pix'] = l_lr


        if getattr(self, 'mean_w', 0.0) and self.mean_w > 0:
            mean_out = _to_gray(output_for_loss).mean(dim=(2, 3))
            mean_gt = _to_gray(self.gt).mean(dim=(2, 3))
            l_mean = torch.mean(torch.abs(mean_out - mean_gt))
            l_total += self.mean_w * l_mean
            loss_dict['l_mean'] = l_mean


        if getattr(self, 'sat_w', 0.0) and self.sat_w > 0 and output_raw is not None:
            over = F.relu(output_raw - 1.0)
            under = F.relu(0.0 - output_raw)
            l_sat = torch.mean(over * over + under * under)
            l_total += self.sat_w * l_sat
            loss_dict['l_sat'] = l_sat
        if self.cri_enhancedIllu:

            ratio = float(getattr(self.net_g, "ratio", 1.0))
            ratio = max(ratio, 1e-6)


            gray_gt = _to_gray(self.gt)
            gray_lq = _to_gray(self.lq)
            gain_gt = torch.clamp(gray_gt / (gray_lq + 1e-3), 0.0, ratio)

            pred_gain = self.enhanced_L
            if pred_gain.dim() == 4 and pred_gain.size(1) == 3:
                pred_gain = _to_gray(pred_gain)


            pred_n = pred_gain / ratio
            tgt_n = gain_gt / ratio

            l_EnhancedIllu = self.cri_enhancedIllu(pred_n, tgt_n)


            warm = int(getattr(self, 'enhancedIllu_warmup_iters', 0) or 0)
            w = 1.0
            if warm > 0:
                w = min(1.0, float(current_iter) / float(warm))
            l_total += w * l_EnhancedIllu
            loss_dict['l_enhancedIllu'] = l_EnhancedIllu

        if getattr(self, 'enhancedIlluTV_w', 0.0) and self.enhancedIlluTV_w > 0:

            ratio = float(getattr(self.net_g, "ratio", 1.0))
            ratio = max(ratio, 1e-6)
            g = self.enhanced_L
            if g.dim() == 4 and g.size(1) == 3:
                g = _to_gray(g)
            g = g / ratio

            tv_h = torch.mean(torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(g[:, :, :, 1:] - g[:, :, :, :-1]))
            l_EnhancedIlluTV = tv_h + tv_w

            warm = int(getattr(self, 'enhancedIlluTV_warmup_iters', 0) or 0)
            w = 1.0
            if warm > 0:
                w = min(1.0, float(current_iter) / float(warm))
            l_total += w * self.enhancedIlluTV_w * l_EnhancedIlluTV
            loss_dict['l_EnhancedIlluTV'] = l_EnhancedIlluTV


        if self.cri_reflRestore:


            with torch.no_grad():
                w = _sobel_grad(_to_gray(self.gt_R))
                w = w / (w.amax(dim=(2, 3), keepdim=True) + 1e-6)
                thr = float(getattr(self, 'refl_edge_thr', 0.15))
                tau = float(getattr(self, 'refl_edge_tau', 0.05))
                base = float(getattr(self, 'refl_base', 0.10))
                w = torch.sigmoid((w - thr) / (tau + 1e-6))
                w = base + (1.0 - base) * w


            l_reflRestore = torch.mean(torch.abs(self.restored_R - self.gt_R) * w)
            refl_w = float(getattr(self.cri_reflRestore, 'loss_weight', 1.0))
            l_reflRestore = refl_w * l_reflRestore


            lf_w = float(getattr(self, 'refl_lf_w', 0.0) or 0.0)
            if lf_w > 0:
                ks = int(getattr(self, 'refl_lf_ks', 31) or 31)
                lf_out = _blur(self.restored_R, ks=ks)
                lf_gt = _blur(self.gt_R, ks=ks)
                l_reflRestore_lf = torch.mean(torch.abs(lf_out - lf_gt))
                l_total += lf_w * l_reflRestore_lf
                loss_dict['l_reflRestore_lf'] = l_reflRestore_lf

            if self.refl_warmup_iters > 0 and current_iter < self.refl_warmup_iters:
                l_reflRestore = l_reflRestore * (float(current_iter) / float(self.refl_warmup_iters))

            l_total += l_reflRestore
            loss_dict['l_reflRestore'] = l_reflRestore

        if self.cri_noisePrior:
            l_noisePrior = self.cri_noisePrior(self.output_noisePrior, self.gt_noisePrior)
            l_total += l_noisePrior
            loss_dict['l_noisePrior'] = l_noisePrior


        if self.cri_edge:

            edge_gt = _sobel_grad(_to_gray(self.gt))
            if getattr(self, 'use_gtmean_edge', False) and (_gtmean_scale is not None) and (_gtmean_wmean is not None):
                edge_out_ori = _sobel_grad(_to_gray(output_for_loss))
                edge_out_align = _sobel_grad(_to_gray(output_for_loss * _gtmean_scale))
                l_edge_img_ori = self.cri_edge(edge_out_ori, edge_gt)
                l_edge_img_align = self.cri_edge(edge_out_align, edge_gt)
                l_edge_img = _gtmean_wmean * l_edge_img_ori + (1.0 - _gtmean_wmean) * l_edge_img_align
                loss_dict['l_edge_img_ori'] = l_edge_img_ori
                loss_dict['l_edge_img_align'] = l_edge_img_align
            else:
                edge_out = _sobel_grad(_to_gray(output_for_loss))
                l_edge_img = self.cri_edge(edge_out, edge_gt)


            r_out = torch.clamp(self.restored_R, 0.0, self.refl_cap)








            if getattr(self, "gt_R", None) is None:
                raise RuntimeError("gt_R is None. Please ensure GT reflectance is provided before computing edge loss.")
            r_gt = torch.clamp(self.gt_R, 0.0, self.refl_cap)

            edge_r_out = _sobel_grad(_to_gray(r_out))
            edge_r_gt = _sobel_grad(_to_gray(r_gt))
            l_edge_r = self.cri_edge(edge_r_out, edge_r_gt)


            l_edge = l_edge_img + 0.7 * l_edge_r


            if self.edge_warmup_iters > 0 and current_iter < self.edge_warmup_iters:
                l_edge = l_edge * (float(current_iter) / float(self.edge_warmup_iters))


            l_total += l_edge
            loss_dict['l_edge'] = l_edge


        if self.cri_hvi:
            hvi_gt = _rgb_to_hvi(self.gt)
            if getattr(self, 'use_gtmean_hvi', False) and (_gtmean_scale is not None) and (_gtmean_wmean is not None):
                hvi_out_ori = _rgb_to_hvi(output_for_loss)
                hvi_out_align = _rgb_to_hvi(output_for_loss * _gtmean_scale)
                l_hvi_ori = self.cri_hvi(hvi_out_ori, hvi_gt)
                l_hvi_align = self.cri_hvi(hvi_out_align, hvi_gt)
                l_hvi = _gtmean_wmean * l_hvi_ori + (1.0 - _gtmean_wmean) * l_hvi_align
                loss_dict['l_hvi_ori'] = l_hvi_ori
                loss_dict['l_hvi_align'] = l_hvi_align
            else:
                hvi_out = _rgb_to_hvi(output_for_loss)
                l_hvi = self.cri_hvi(hvi_out, hvi_gt)

            if self.hvi_warmup_iters > 0 and current_iter < self.hvi_warmup_iters:
                l_hvi = l_hvi * (float(current_iter) / float(self.hvi_warmup_iters))

            l_total += l_hvi
            loss_dict['l_hvi'] = l_hvi


        chroma_w = float(getattr(self, 'chroma_w', 0.0) or 0.0)
        if chroma_w > 0:
            out01 = output_for_loss.clamp(0, 1)
            gt01 = self.gt.clamp(0, 1)
            _, cb_o, cr_o = _rgb_to_ycbcr(out01)
            _, cb_g, cr_g = _rgb_to_ycbcr(gt01)
            l_chroma = (cb_o - cb_g).abs().mean() + (cr_o - cr_g).abs().mean()

            if getattr(self, 'chroma_warmup_iters', 0) > 0 and current_iter < self.chroma_warmup_iters:
                l_chroma = l_chroma * (float(current_iter) / float(self.chroma_warmup_iters))

            l_total += chroma_w * l_chroma
            loss_dict['l_chroma'] = l_chroma


        if getattr(self, 'wb_reg_w', 0.0) and self.wb_reg_w > 0 and (getattr(self, 'wb_gain', None) is not None):
            l_wb = torch.mean(torch.abs(self.wb_gain - 1.0))
            l_total += self.wb_reg_w * l_wb
            loss_dict['l_wb_gain'] = l_wb


        l_total.backward()
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):


        use_ema = hasattr(self, 'net_g_ema') and (self.net_g_ema is not None)
        net = self.net_g_ema if use_ema else self.net_g


        was_training = net.training
        net.eval()

        with torch.no_grad():
            out_net = net(self.lq)
            (self.output_test,
             self.highL_test,
             self.L_test,
             self.restoredR_test,
             self.R_test,
             self.noise_test,
             self.L_prior_cond_test) = out_net[:7]


            self.extra_test_outputs = out_net[7:] if len(out_net) > 7 else None


            self.output_test = torch.clamp(self.output_test, 0.0, 1.0)

            val_opt = self.opt.get('val', {})
            vis_exp = float(val_opt.get('vis_exposure', 1.25))
            vis_gam = float(val_opt.get('vis_gamma', 0.85))
            out = self.output_test
            out_vis = torch.clamp((out.clamp(0, 1).pow(vis_gam)) * vis_exp, 0.0, 1.0)
            self.output_vis_test = out_vis


            self.highL_raw = self.highL_test.detach()
            self.L_raw = self.L_test.detach()
            self.R_raw = self.R_test.detach()
            self.restoredR_raw = self.restoredR_test.detach()



            gain = getattr(net, "gain_L_last", None)

            if gain is not None:
                ratio = float(getattr(net, "ratio", 7.0))
                illu_g = float(getattr(net, "illu_gamma", 0.7))


                high_illu = (self.L_raw * gain).clamp(0.0, ratio)
                high_illu = high_illu.clamp_min(1e-6).pow(illu_g)


                self.highL_test = _vis01(high_illu, k=2.5).pow(0.6)
            else:

                self.highL_test = _vis01(torch.clamp(self.highL_raw, 0.0, 1.0), k=2.5).pow(0.6)

            self.L_test = _norm01(self.L_raw)
            self.R_test = _norm01(self.R_raw)
            self.restoredR_test = _norm01(self.restoredR_raw)


            s_raw = _sobel_grad(_to_gray(self.lq))
            self.s_test = _norm01(s_raw)


            if hasattr(self, "gt") and (self.gt is not None):
                try:
                    out_gt = net(self.gt)
                    self.gt_L_test = out_gt[2]
                    self.gt_R_test = out_gt[4]
                    self.gt_noise_test = out_gt[5]

                    self.gt_L_test = _norm01(self.gt_L_test.detach())
                    self.gt_R_test = _norm01(self.gt_R_test.detach())
                    self.gt_noise_test = _norm01(self.gt_noise_test.detach())
                except Exception:

                    pass


            def _stat(x, name):
                x = x.detach()
                return f"{name}: min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f}"

            if not hasattr(self, "_dbg_printed_once"):
                print(_stat(self.highL_raw, "HighL_raw"),
                      _stat(self.L_raw, "L_raw"),
                      _stat(self.R_raw, "R_raw"),
                      _stat(self.restoredR_raw, "restoredR_raw"))
                self._dbg_printed_once = True



        if use_ema:

            self.net_g.train()

            self.net_g_ema.eval()
        else:
            if was_training:
                self.net_g.train()
            else:
                self.net_g.eval()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']


        val_opt = self.opt.get('val', {})
        calc_lpips = bool(val_opt.get('calc_lpips', False))
        calc_niqe = bool(val_opt.get('calc_niqe', False))
        lpips_net = val_opt.get('lpips_net', 'alex')
        metric_device = str(val_opt.get('metric_device', 'cpu')).lower().strip()
        if metric_device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1']:
            metric_device = 'cpu'

        base_metrics_opt = val_opt.get('metrics', None)
        has_base_metrics = base_metrics_opt is not None
        with_metrics = bool(has_base_metrics or calc_lpips or calc_niqe)

        if with_metrics:
            self.metric_results = {}
            if has_base_metrics:
                self.metric_results.update({metric: 0 for metric in base_metrics_opt.keys()})
            if calc_lpips:
                self.metric_results['lpips'] = 0
            if calc_niqe:
                self.metric_results['niqe'] = 0
            metric_data = dict()


        _ = val_opt.get('suffix', '')

        pbar = tqdm(total=len(dataloader), unit='image')


        lpips_metric = None
        niqe_metric = None

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            enhanced_img = tensor2img([visuals['enhanced']])
            enhanced_vis_img = tensor2img([visuals.get('enhanced_vis', visuals['enhanced'])])


            if save_img:
                save_dir = osp.join(self.opt['path']['visualization'], img_name)
                os.makedirs(save_dir, exist_ok=True)

                imwrite(enhanced_img, osp.join(save_dir, f'{img_name}_{current_iter}_enhanced.png'))


                if 'highL' in visuals:
                    highL_img = tensor2img([visuals['highL']])
                    imwrite(highL_img, osp.join(save_dir, f'{img_name}_{current_iter}_highL.png'))
                if 'restoredR' in visuals:
                    restoredR_img = tensor2img([visuals['restoredR']])
                    imwrite(restoredR_img, osp.join(save_dir, f'{img_name}_{current_iter}_restoredR.png'))
                if 's' in visuals:
                    s_img = tensor2img([visuals['s']])
                    imwrite(s_img, osp.join(save_dir, f'{img_name}_{current_iter}_s.png'))
                if 'L' in visuals:
                    L_img = tensor2img([visuals['L']])
                    imwrite(L_img, osp.join(save_dir, f'{img_name}_{current_iter}_L.png'))
                if 'R' in visuals:
                    R_img = tensor2img([visuals['R']])
                    imwrite(R_img, osp.join(save_dir, f'{img_name}_{current_iter}_R.png'))


















                lq_img = None
                if 'lq' in visuals:
                    lq_img = tensor2img([visuals['lq']])
                elif 'lq' in val_data:
                    lq_img = tensor2img([val_data['lq']])


                gt_img = None
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                elif 'gt' in val_data:
                    gt_img = tensor2img([val_data['gt']])



                imgs_2x4 = [
                    lq_img,
                    gt_img,
                    enhanced_vis_img,
                    locals().get('s_img', None),
                    locals().get('L_img', None),

                    locals().get('highL_img', None),
                    locals().get('R_img', None),
                    locals().get('restoredR_img', None),

                ]

                titles_2x4 = [
                    "Input (LQ)",
                    "GT",
                    "Output(Ours)",
                    "Structure",
                    "Illumination (L)",

                    "High-L",
                    "Reflectance (R)",
                    "Restored R",

                ]

                grid_pil = make_grid_2x4_with_titles_center(
                    imgs_2x4, titles_2x4,
                    pad=10, title_h=50, font_size=24
                )

                if grid_pil is not None:
                    grid_np = np.array(grid_pil)[:, :, ::-1]
                    imwrite(grid_np, osp.join(save_dir, f'{img_name}_{current_iter}_visgrid_2x4.png'))


            if with_metrics:
                metric_data['img'] = enhanced_img
                gt_img = None
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                    metric_data['img2'] = gt_img
                    if hasattr(self, 'gt'):
                        del self.gt


                if has_base_metrics:
                    for name, opt_ in base_metrics_opt.items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)


                if calc_lpips and (gt_img is not None):
                    try:
                        if lpips_metric is None:
                            import lpips as _lp
                            lpips_metric = _lp.LPIPS(net=lpips_net).eval().to(metric_device)
                        x = _np_uint8_bgr_to_torch_rgb01(enhanced_img).to(metric_device)
                        y = _np_uint8_bgr_to_torch_rgb01(gt_img).to(metric_device)
                        with torch.no_grad():
                            v = lpips_metric(_torch01_to_lpips_input(x), _torch01_to_lpips_input(y))
                        self.metric_results['lpips'] += float(v.mean().item())
                    except Exception as e:
                        warnings.warn(f'[val] LPIPS skipped: {e}')


                if calc_niqe:
                    try:
                        if niqe_metric is None:
                            import pyiqa as _pyiqa
                            niqe_metric = _pyiqa.create_metric('niqe').eval().to(metric_device)
                        x = _np_uint8_bgr_to_torch_rgb01(enhanced_img).to(metric_device)
                        with torch.no_grad():
                            n = niqe_metric(x)
                        self.metric_results['niqe'] += float(n.mean().item())
                    except Exception as e:
                        warnings.warn(f'[val] NIQE skipped: {e}')

            pbar.update(1)
            pbar.set_description(f'Val {img_name}')

        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'	{metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict['enhanced'] = self.output_test.detach().cpu()
        if hasattr(self, "output_vis_test"):
            out_dict['enhanced_vis'] = self.output_vis_test.detach().cpu()


        if hasattr(self, 'highL_test'):
            out_dict['highL'] = self.highL_test.detach().cpu()
        if hasattr(self, 'restoredR_test'):
            out_dict['restoredR'] = self.restoredR_test.detach().cpu()
        if hasattr(self, 's_test'):
            out_dict['s'] = self.s_test.detach().cpu()
        if hasattr(self, 'L_test'):
            out_dict['L'] = self.L_test.detach().cpu()
        if hasattr(self, 'R_test'):
            out_dict['R'] = self.R_test.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)


        if hasattr(self, 'net_noisePrior') and self.net_noisePrior is not None:
            self.save_network(self.net_noisePrior, 'net_noisePrior', current_iter)

        self.save_training_state(epoch, current_iter)
