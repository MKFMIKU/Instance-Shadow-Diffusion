from os.path import join, basename
import copy

import time

from tqdm.auto import tqdm

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from accelerate import Accelerator
from accelerate.utils import set_seed

from kornia.morphology import dilation

from basicsr.data.transforms import augment
from basicsr.data.data_util import paired_paths_from_folder
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img

from guided_diffusion.script_util import create_gaussian_diffusion

from diffusion_arch import DensePosteriorConditionalUNet

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BATCH_SIZE = 2
TOTAL_ITERS = 800000
ACCUM = 16


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in target_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=ACCUM,
        mixed_precision='fp16',
        project_dir="experiments",
    )
    set_seed(10666)

    if accelerator:
        print = accelerator.print

    print(f"=> Inited Accelerator")

    eval_dataset = ISTDDataset(opt={
        'io_backend': {'type': 'disk'},
        'dataroot': 'data/aistd_test',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'use_rot': True,
        'use_hflip': True,
        'interval': 50,
        'phase': 'test'
    })
    print(f"=> Loaded dataset with number of {len(eval_dataset)} for testing")


    model = DensePosteriorConditionalUNet(
        in_channels=3 + 3 + 1,
        out_channels=3,
        model_channels=192,
        num_res_blocks=2,
        attention_resolutions=[8, 16, 32],
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        channel_mult=[1, 1, 2, 2, 2, 4],
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    ema_model = copy.deepcopy(model)

    feature_encoder = DensePosteriorConditionalUNet(
        in_channels=3 + 3,
        out_channels=1,
        model_channels=96,
        num_res_blocks=1,
        attention_resolutions=[8, 16],
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        channel_mult=[1, 1, 2, 2, 4],
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    ema_feature_encoder = copy.deepcopy(feature_encoder)

    print(f"=> Inited optimizer")

    device = accelerator.device
    ema_model.to(device)
    ema_feature_encoder.to(device)

    model, ema_model, feature_encoder, ema_feature_encoder = accelerator.prepare(
        model, ema_model, feature_encoder, ema_feature_encoder
    )


    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=False,
        noise_schedule='linear',
        use_kl=False,
        timestep_respacing="ddim50",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=0.5,
        p2_k=1,
    )

    accelerator.load_state('experiments/state_059999.bin')

    avg_psnr = []
    for i in tqdm(range(len(eval_dataset)), leave=True):
        batch = eval_dataset[i]
        gt = batch['gt'].unsqueeze(0).to(device, non_blocking=True)
        lq = batch['lq'].unsqueeze(0).to(device, non_blocking=True)
        mask = batch['mask'].unsqueeze(0).to(device, non_blocking=True)
        mask = dilation(mask, torch.ones(21, 21).to(mask.device))


        with torch.no_grad():
            intrinsic_feature = ema_feature_encoder(lq, torch.tensor([0], device=device), latent=mask)

            start_time = time.time()
            latent = torch.cat((lq, intrinsic_feature), dim=1)
            pred_gt = eval_gaussian_diffusion.ddim_sample_loop(ema_model, lq.shape, model_kwargs={'latent': latent}, progress=True)
            print("--- %s seconds ---" % (time.time() - start_time))

            pred_gt = pred_gt.clip(-1, 1)
            pred_gt = 255. * (pred_gt / 2 + 0.5)
            gt = 255. * (gt / 2 + 0.5)
            psnr = 20 * torch.log10(255.0 / torch.sqrt(torch.mean((pred_gt - gt) ** 2)))
            intrinsic_feature = 255. * (intrinsic_feature.repeat((1, 3, 1, 1)) / 2 + .5)

            print(f"===> psnr: {psnr:05f}")
            avg_psnr.append(psnr)

            cv2.imwrite(f"experiments/exp_{i:06d}.png", tensor2img(torch.cat((gt, intrinsic_feature, pred_gt), dim=-1), min_max=(0, 255.)))

        print(f"testing_psnr {sum(avg_psnr) / len(avg_psnr)}")


class ISTDDataset(data.Dataset):
    def __init__(self, opt):
        super(ISTDDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.interval = opt['interval']

        self.gt_folder, self.lq_folder = join(opt['dataroot'], 'shadow_free'), join(opt['dataroot'], 'shadow')
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        mask_path = join(self.opt['dataroot'], 'mask', basename(lq_path))
        img_bytes = self.file_client.get(mask_path, 'mask')
        img_mask = imfrombytes(img_bytes, float32=True)

        img_gt = cv2.resize(img_gt, (256, 256), cv2.INTER_CUBIC)
        img_lq = cv2.resize(img_lq, (256, 256), cv2.INTER_CUBIC)
        img_mask = cv2.resize(img_mask, (256, 256), cv2.INTER_NEAREST)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # flip, rotation
            img_gt, img_lq, img_mask = augment([img_gt, img_lq, img_mask], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'mask': img_mask, 'gt_path': lq_path, 'gt_idx': index}

    def __len__(self):
        return int(len(self.paths) // self.interval)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
