import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import itertools
import os
import time
import argparse
import json
from pathlib import Path

from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from utils import AttrDict, build_env, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def train(h, args):
    torch.manual_seed(h.seed)
    device = torch.device('cuda:0')
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Optimizers
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    
    # Load checkpoint if exists
    steps = 0
    last_epoch = -1
    
    if args.resume:
        cp_g = scan_checkpoint(checkpoint_dir, 'g_')
        cp_do = scan_checkpoint(checkpoint_dir, 'do_')
        
        if cp_g is not None and cp_do is not None:
            print(f"Resuming from checkpoint: {cp_g}")
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            
            generator.load_state_dict(state_dict_g['generator'])
            optim_g.load_state_dict(state_dict_g['optim_g'])
            steps = state_dict_g['steps'] + 1
            last_epoch = state_dict_g['epoch']
            
            mpd.load_state_dict(state_dict_do['mpd'])
            msd.load_state_dict(state_dict_do['msd'])
            optim_d.load_state_dict(state_dict_do['optim_d'])
            
            print(f"✓ Resumed from step {steps}, epoch {last_epoch}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    
    # Data loaders
    training_filelist, validation_filelist = get_dataset_filelist(args)
    
    trainset = MelDataset(
        training_filelist, h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
        n_cache_reuse=0, shuffle=True, fmax_loss=h.fmax_for_loss,
        device=device, fine_tuning=False
    )
    
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=True,
        batch_size=h.batch_size, pin_memory=True, drop_last=True
    )
    
    validset = MelDataset(
        validation_filelist, h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
        False, False, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
        device=device, fine_tuning=False
    )
    
    val_loader = DataLoader(
        validset, num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True, drop_last=True
    )
    
    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    sw = SummaryWriter(os.path.join(checkpoint_dir, 'logs'))
    
    # Training loop
    generator.train()
    mpd.train()
    msd.train()
    
    print("Starting training...")
    print(f"Checkpoint will be saved every {args.checkpoint_interval} steps")
    print(f"Training for {args.training_steps} total steps")
    
    start_time = time.time()
    
    for epoch in range(max(0, last_epoch), args.training_epochs):
        print(f"\\nEpoch: {epoch+1}")
        
        for i, batch in enumerate(train_loader):
            if steps >= args.training_steps:
                print(f"\\n✓ Reached target steps ({args.training_steps})")
                break
                
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)
            
            # Generator
            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
            )
            
            optim_d.zero_grad()
            
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()
            
            # Generator
            optim_g.zero_grad()
            
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            
            loss_gen_all.backward()
            optim_g.step()
            
            # Logging
            if steps % args.stdout_interval == 0:
                elapsed = time.time() - start_time
                print(f"Steps: {steps:7d} | Mel: {loss_mel.item():.4f} | "
                      f"Gen: {loss_gen_all.item():.4f} | Disc: {loss_disc_all.item():.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            if steps % args.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", loss_mel, steps)
                sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)
            
            # Validation
            if steps % args.validation_interval == 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        if j >= 4:  # Only validate on 4 samples
                            break
                        x, y, _, y_mel = batch
                        y_g_hat = generator(x.to(device))
                        y_mel = y_mel.to(device, non_blocking=True)
                        y_g_hat_mel = mel_spectrogram(
                            y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                            h.sampling_rate, h.hop_size, h.win_size,
                            h.fmin, h.fmax_for_loss
                        )
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                
                val_err = val_err_tot / 4
                sw.add_scalar("validation/mel_spec_error", val_err, steps)
                print(f"Validation mel error: {val_err:.4f}")
                
                generator.train()
            
            # Checkpointing
            if steps % args.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'g_{steps:08d}')
                save_checkpoint(
                    checkpoint_path,
                    {'generator': generator.state_dict(),
                     'optim_g': optim_g.state_dict(),
                     'steps': steps,
                     'epoch': epoch}
                )
                
                checkpoint_path = os.path.join(checkpoint_dir, f'do_{steps:08d}')
                save_checkpoint(
                    checkpoint_path,
                    {'mpd': mpd.state_dict(),
                     'msd': msd.state_dict(),
                     'optim_d': optim_d.state_dict(),
                     'steps': steps,
                     'epoch': epoch}
                )
                
                print(f"✓ Checkpoint saved at step {steps}")
            
            steps += 1
        
        scheduler_g.step()
        scheduler_d.step()
        
        if steps >= args.training_steps:
            break
    
    print("\\n✓ Training completed!")
    sw.close()


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--training_steps', default=500000, type=int)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--stdout_interval', default=50, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--input_training_file', required=True)
    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--input_wavs_dir', required=True)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    build_env(args.config, 'config.json', args.checkpoint_path)
    
    train(h, args)