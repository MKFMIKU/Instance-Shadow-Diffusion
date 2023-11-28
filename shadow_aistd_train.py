from os.path import join, basename
from tqdm.auto import tqdm
import copy

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

BATCH_SIZE = 8
LR = 2e-5
TOTAL_ITERS = 800_000
ACCUM = 1

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
        log_with="tensorboard",
    )
    accelerator.init_trackers("instance_shadow_diffusion")
    set_seed(10666)

    if accelerator:
        print = accelerator.print

    print(f"=> Inited Accelerator")

    dataset = ISTDDataset(opt={
        'io_backend': {'type': 'disk'},
        'dataroot': 'data/aistd_train',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'use_rot': True,
        'use_hflip': True,
        'interval': 1,
        'phase': 'train'
    })

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
    print(f"=> Loaded dataset with number of {len(dataset)} for training and {len(eval_dataset)} for testing")

    def loopy(dl):
        while True:
            for x in iter(dl):
                yield x

    dataloader = loopy(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
    )

    model = DensePosteriorConditionalUNet(
        in_channels=3 + 3 + 1,
        out_channels=6,
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

    optim_shadow = torch.optim.AdamW(
        [{"params": model.parameters()}, {"params": feature_encoder.parameters()}],
        lr=LR,
        weight_decay=0.,
    )
    optim_refine_shadow = torch.optim.AdamW(
        [{"params": model.parameters()}],
        lr=LR,
        weight_decay=0.,
    )


    print(f"=> Inited optimizer")

    device = accelerator.device
    model.to(device)
    ema_model.to(device)
    feature_encoder.to(device)
    ema_feature_encoder.to(device)

    model, ema_model, feature_encoder, ema_feature_encoder, optim_shadow, optim_refine_shadow, dataloader = accelerator.prepare(
        model, ema_model, feature_encoder, ema_feature_encoder, optim_shadow, optim_refine_shadow, dataloader
    )

    gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=1,
        p2_k=1,
    )

    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        timestep_respacing="ddim20",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=1,
        p2_k=1,
    )

    print("=> Begin training")
    with tqdm(total=TOTAL_ITERS, disable=not accelerator.is_local_main_process) as pbar:
        for total_iter in range(TOTAL_ITERS):
            batch = next(iter(dataloader))

            gt = batch['gt'].to(device, non_blocking=True)
            lq = batch['lq'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            mask = dilation(mask, torch.ones(21, 21).to(mask.device))

            diffusion_t = np.random.choice(1000, size=(BATCH_SIZE,))
            diffusion_t = torch.from_numpy(diffusion_t).to(device).long()

            intrinsic_feature = feature_encoder(lq, diffusion_t * 0, latent=mask)
            latent = torch.cat((lq, intrinsic_feature), dim=1)

            loss = gaussian_diffusion.training_losses(model, gt, diffusion_t, model_kwargs={'latent': latent})
            loss = torch.mean(loss["loss"])

            accelerator.backward(loss)

            if total_iter < 100_000:
                optim_shadow.step()
                optim_shadow.zero_grad()
            else:
                optim_refine_shadow.step()
                optim_refine_shadow.zero_grad()

            if accelerator.sync_gradients:
                ema(model, ema_model, 0.999)
                ema(feature_encoder, ema_feature_encoder, 0.999)
                if (total_iter + 1) % (ACCUM*500) == 0:
                    if accelerator.is_main_process:
                        avg_psnr = []
                        for i in tqdm(range(len(eval_dataset)), leave=False):
                            batch = eval_dataset[i]
                            gt = batch['gt'].unsqueeze(0).to(device, non_blocking=True)
                            lq = batch['lq'].unsqueeze(0).to(device, non_blocking=True)
                            mask = batch['mask'].unsqueeze(0).to(device, non_blocking=True)
                            mask = dilation(mask, torch.ones(21, 21).to(mask.device))

                            with torch.no_grad():
                                intrinsic_feature = ema_feature_encoder(lq, torch.tensor([0], device=device), latent=mask)
                                latent = torch.cat((lq, intrinsic_feature), dim=1)
                                pred_gt = eval_gaussian_diffusion.ddim_sample_loop(ema_model, lq.shape, model_kwargs={'latent': latent}, progress=False)

                            pred_gt = pred_gt.clip(-1, 1)
                            pred_gt = 255. * (pred_gt / 2 + 0.5)
                            lq = 255. * (lq / 2 + 0.5)
                            mask = 255. * (mask / 2 + 0.5)
                            gt = 255. * (gt / 2 + 0.5)
                            psnr = 20 * torch.log10(255.0 / torch.sqrt(torch.mean((pred_gt - gt) ** 2)))
                            avg_psnr.append(psnr)

                        accelerator.log({"testing_psnr": sum(avg_psnr) / len(avg_psnr)}, step=total_iter)
                        cv2.imwrite(f"experiments/exp_{total_iter:06d}.png", tensor2img(torch.cat((lq, mask, pred_gt, gt), dim=-1), min_max=(0, 255.)))

                if (total_iter + 1) % (ACCUM*5000) == 0:
                    if accelerator.is_main_process:
                        accelerator.save_state(f"experiments/state_{total_iter:06d}.bin")

            accelerator.log({"training_loss": loss}, step=total_iter)
            pbar.set_description(
                f"Logs Iter: {total_iter:06d} Loss {loss.item():.5f}"
            )
            pbar.update(1)



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
