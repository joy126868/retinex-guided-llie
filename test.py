import os, argparse
import torch
import numpy as np
from PIL import Image

from basicsr.utils.options import parse_options
from basicsr.utils import get_root_logger, get_env_info, make_exp_dirs
from basicsr.data import build_dataset, build_dataloader
from basicsr.models import build_model
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    """Tensor -> uint8 HWC RGB。支援 [B,C,H,W] / [C,H,W] / 單通道。"""
    if x.dim() == 4:
        x = x[0]
    assert x.dim() == 3, f'Expect [C,H,W], got {x.shape}'
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    elif c > 3:
        x = x[:3, ...]
    x = x.clamp(0, 1).permute(1, 2, 0).contiguous()  # [H,W,3]
    return (x.cpu().numpy() * 255.0).round().astype(np.uint8)


@torch.no_grad()
def _net_forward_rgb(model, lq: torch.Tensor) -> torch.Tensor:
    """只取 net 輸出的第一個（最終增亮影像）。優先使用 EMA。"""
    net = getattr(model, 'net_g_ema', None) or model.net_g
    net.eval()
    out_all = net(lq)
    return out_all[0] if isinstance(out_all, (list, tuple)) else out_all


def test_pipeline(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML')
    parser.add_argument('--ckpt', type=str, default='', help='Override G checkpoint path')
    args = parser.parse_args()

    # 1) 解析 YAML & 覆蓋設定
    opt, _ = parse_options(root_path, is_train=False)
    opt.setdefault('path', {})
    opt['path']['strict_load_g'] = False
    if args.ckpt:
        opt['path']['pretrain_network_g'] = args.ckpt
    print('CKPT =>', opt['path'].get('pretrain_network_g'))

    # 2) log
    make_exp_dirs(opt)
    logger = get_root_logger(logger_name='basicsr', log_level='INFO')
    logger.info(get_env_info())

    # 3) dataloaders（優先 test/val）
    test_loaders, picked = [], False
    for key in ('test', 'val'):
        if key in opt['datasets']:
            dopt = opt['datasets'][key]
            dset = build_dataset(dopt)
            dloader = build_dataloader(
                dset, dopt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=None, seed=opt['manual_seed'])
            logger.info(f"Dataset[{key}] {dopt['name']} -> {len(dset)} images")
            test_loaders.append(dloader)
            picked = True
    if not picked:
        for _, dopt in sorted(opt['datasets'].items()):
            dset = build_dataset(dopt)
            dloader = build_dataloader(
                dset, dopt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=None, seed=opt['manual_seed'])
            logger.info(f"Dataset[*] {dopt['name']} -> {len(dset)} images")
            test_loaders.append(dloader)

    # 4) model
    model = build_model(opt)
    net = getattr(model, 'net_g_ema', None) or model.net_g
    device = next(net.parameters()).device

    # 5) 推論＋評分
    for test_loader in test_loaders:
        name = test_loader.dataset.opt['name']
        save_dir = os.path.join('results', opt['name'], 'visualization', name)
        os.makedirs(save_dir, exist_ok=True)
        print(f'\nTesting {name} -> {save_dir}')

        # 與 YAML 的 val.metrics 對齊（若沒設，採預設）
        use_y = opt.get('val', {}).get('metrics', {}).get('psnr', {}).get('test_y_channel', False)
        crop_border = opt.get('val', {}).get('metrics', {}).get('psnr', {}).get('crop_border', 0)

        tot_psnr, tot_ssim, n = 0.0, 0.0, 0

        for data in test_loader:
            lq = data['lq'].to(device, non_blocking=True).clamp(0, 1)
            out = _net_forward_rgb(model, lq)                 # [B,C,H,W]
            pred_np = _to_uint8_hwc(out)

            # 檔名
            if 'lq_path' in data and len(data['lq_path']) > 0:
                stem = os.path.splitext(os.path.basename(data['lq_path'][0]))[0]
            elif 'gt_path' in data and len(data['gt_path']) > 0:
                stem = os.path.splitext(os.path.basename(data['gt_path'][0]))[0]
            else:
                stem = f'{n:06d}'

            # 存圖
            Image.fromarray(pred_np).save(os.path.join(save_dir, f'{stem}_enhanced.png'))

            # 若 dataloader 有 GT，計分
            if 'gt' in data and data['gt'] is not None:
                gt = data['gt'].to(device, non_blocking=True).clamp(0, 1)
                if out.shape[-2:] != gt.shape[-2:]:
                    gt = torch.nn.functional.interpolate(gt, size=out.shape[-2:], mode='bilinear', align_corners=True)
                gt_np = _to_uint8_hwc(gt)

                psnr = calculate_psnr(pred_np, gt_np, crop_border=crop_border, test_y_channel=use_y)
                ssim = calculate_ssim(pred_np, gt_np, crop_border=crop_border, test_y_channel=use_y)
                tot_psnr += psnr
                tot_ssim += ssim
                n += 1

        if n > 0:
            print(f' PSNR: {tot_psnr / n:.3f} dB, \n SSIM: {tot_ssim / n:.4f} '
                  )

    print('\nDONE.')


if __name__ == '__main__':
    root_path = os.path.abspath('.')
    test_pipeline(root_path)






# import logging
# import torch
# from os import path as osp
# import os
#
# from basicsr.data import build_dataloader, build_dataset
# from basicsr.models import build_model
# from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
# from basicsr.utils.options import dict2str, parse_options
# from PIL import Image
# import torchvision.transforms.functional as TF
#
#
#
#
# def test_pipeline(root_path):
#     # parse options, set distributed setting, set ramdom seed
#     opt, _ = parse_options(root_path, is_train=False)
#
#     torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.deterministic = True
#
#     # mkdir and initialize loggers
#     make_exp_dirs(opt)
#     log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))
#
#     # create test dataset and dataloader
#     test_loaders = []
#     for _, dataset_opt in sorted(opt['datasets'].items()):
#         test_set = build_dataset(dataset_opt)
#         test_loader = build_dataloader(
#             test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#         logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
#         test_loaders.append(test_loader)
#
#     # create model
#     model = build_model(opt)
#
#     for test_loader in test_loaders:
#         test_set_name = test_loader.dataset.opt['name']
#         logger.info(f'Testing {test_set_name}...')
#         model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
#
#
# if __name__ == '__main__':
#     # root_path = r"D:/CUE-master"
#     root_path = os.path.abspath('.')
#
#
#     test_pipeline(root_path)
