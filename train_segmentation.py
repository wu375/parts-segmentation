import numpy as np
import scipy
import os
import pickle
import argparse
import random
from itertools import repeat
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from tensorboardX import SummaryWriter

from video_datasets import ClevrerDataset, SpmotDataset
from parts import Parts

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_logging():
    os.makedirs("train_log", exist_ok=True)
    i = 0
    while os.path.isdir('train_log/run'+str(i)):
        i += 1
    log_dir = 'train_log/run'+str(i)+'/'
    return log_dir

def get_grad_norm(model):
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** 0.5
    return grad_norm

def inf_loader_wrapper(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

seven_colors_map = torch.tensor([
    [148, 0, 211],
    [75, 0, 130],
    [0, 0, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 127, 0],
    [255, 0 , 0],
])

def log_stuff(writer, results, step, tag='train'):
    if tag == 'val':
        video_batch = results.pop('recons', None)[:1]
        writer.add_video(f'recon_' + tag, video_batch.detach().cpu().numpy(), step, fps=10)

        video_batch = results.pop('recons_slots', None)[:1]

        # writer.add_video(f'recon_' + tag, video_batch.detach().cpu().numpy(), step, fps=10)
        for k in range(opt.K):
            writer.add_video(f'recon_{k}_' + tag, video_batch[:, :, k].detach().cpu().numpy(), step, fps=10)


        video_batch = results.pop('gt', None)[:1]
        writer.add_video('gt_' + tag, video_batch.detach().cpu().numpy(), step, fps=10)

        video_batch = results.pop('seg_topdown', None)[:1] # (1, time, w, h)
        video_batch = seven_colors_map[video_batch] # (1, time, w, h, 3)
        video_batch = video_batch.permute(0, 1, 4, 2, 3)
        writer.add_video('seg_topdown_' + tag, video_batch.detach().cpu().numpy(), step, fps=10)

        video_batch = results.pop('seg_bottomup', None)[:1] # (1, time, w, h)
        video_batch = seven_colors_map[video_batch] # (1, time, w, h, 3)
        video_batch = video_batch.permute(0, 1, 4, 2, 3)
        writer.add_video('seg_bottomup_' + tag, video_batch.detach().cpu().numpy(), step, fps=10)

    for key in results:
        if isinstance(results[key], torch.Tensor):
            writer.add_scalar(key + '_' + tag, results[key].detach().mean().cpu().item(), step)
        else:
            writer.add_scalar(key + '_' + tag, results[key], step)

def train():
    set_seeds(opt.seed)
    log_dir = setup_logging()
    print('log_dir is ', log_dir)

    task_name = 'parts_seg'

    model = Parts(
        K=opt.K, # aka n_slots
        D_slot=64,
        height=opt.image_size,
        width=opt.image_size,
        n_refine_steps=1,
        transformer_d_model=1024,
        decoder_var=0.09,
        decoder_hidden=64,
        rnn_hidden=256,
        encoder_hidden=64,
        encoder_channel=128, 
        beta=0.3,
        mode='segmentation',
    )
    # model.load('train_log/run18/parts_parts_seg_53175.pth')
    model.cuda()

    if opt.dataset == 'spmot':
        train_dataset = SpmotDataset(seq_len=10, is_train=True, size=opt.image_size)
    elif opt.dataset == 'clevrer':
        train_dataset = ClevrerDataset(seq_len=25, is_train=True, size=opt.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=16,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2**32 + id),
    )

    train_loader = inf_loader_wrapper(train_loader)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)

    writer = SummaryWriter(log_dir)
    try:
        for step in tqdm(range(opt.n_steps)):
            obs = next(train_loader) # (batch, time, c, w, h)

            obs = obs.cuda()
            model.train()

            out = model.forward(obs, optimizer=optimizer, detach_period=25)

            loss = out['loss']

            if step % opt.val_steps == 0:
                results = {}
                results['gt'] = obs
                results['recons'] = out['recons']
                results['recons_slots'] = out['recons_slots']
                results['seg_topdown'] = out['seg_topdown']
                results['seg_bottomup'] = out['seg_bottomup']
                log_stuff(writer, results, step, tag='val')

            if step % opt.log_interval == 0:
                results = out
                results['mse'] = F.mse_loss(results['recons'], obs)
                results['lamb_min'] = results['lamb'].min()
                results['lamb_max'] = results['lamb'].max()
                results['lamb_mean'] = results['lamb'].mean()
                results.pop('lamb', None)
                results.pop('hidden', None)
                results.pop('recons', None)
                results.pop('recons_slots', None)
                results.pop('seg_topdown', None)
                results.pop('seg_bottomup', None)
                results['grad_norm'] = get_grad_norm(model)
                log_stuff(writer, results, step)

            if step != 0 and step % opt.checkpoint_interval == 0:
                model.save(os.path.join(log_dir, f"parts_{task_name}_{step}.pth"))

        model.save(os.path.join(log_dir, f'parts_{task_name}_{step}.pth'))
    except KeyboardInterrupt:
    # except:
        model.save(os.path.join(log_dir, f'parts_{task_name}_{step}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=400)
    parser.add_argument("--checkpoint_interval", type=int, default=20000)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=40)
    parser.add_argument('--dataset', type=str, default='spmot')
    parser.add_argument("--K", type=int, default=4)
    opt = parser.parse_args()
    train()