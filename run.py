import os
import torch
from engine.dehaze import train
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from model.base import CLCC
from utils.common_utils import parse_yaml, Logger, print_params_and_macs, save_dict_as_yaml
from torch.cuda.amp import GradScaler


def configuration_dataloader(hparams, stage_index):
    """Configure dataloaders with training augmentation for the given stage."""
    train_dataset = UIEBTrain(
        folder=hparams['data']['train_path'],
        size=hparams['data']['train_img_size'][stage_index]
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory'],
        drop_last=True  # Ensures consistent batch sizes
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,  # Important: no shuffling for validation
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def configuration_dataloader_no_aug(hparams, stage_index):
    """Configure dataloaders without training augmentation (uses validation transforms for both)."""
    train_dataset = UIEBValid(
        folder=hparams['data']['train_path'],
        size=256
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory'],
        drop_last=True
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def configuration_optimizer(model, hparams):
    """Configure optimizer and learning rate scheduler."""
    optimizer = AdamW(
        params=model.parameters(),
        lr=hparams['optim']['lr_init'],
        weight_decay=hparams['optim']['weight_decay']
    )
    
    # Calculate scheduler parameters
    total_epochs = sum(hparams['train']['stage_epochs'])
    num_stages = len(hparams['train']['stage_epochs'])
    
    if hparams['optim']['use_cycle_limit']:
        # Cycle restarts at each stage
        t_initial = total_epochs // num_stages
        cycle_limit = num_stages
    else:
        # Single cycle for entire training
        t_initial = total_epochs
        cycle_limit = 1
    
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=t_initial,
        lr_min=hparams['optim']['lr_min'],
        cycle_limit=cycle_limit,
        cycle_decay=hparams['optim']['cycle_decay'],
        warmup_t=hparams['optim']['warmup_epochs'],
        warmup_lr_init=hparams['optim']['lr_min']
    )
    return optimizer, scheduler


def main():
    """Main training function with stage-wise training loop."""
    # Set device
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    
    # Load configuration
    args = parse_yaml(r'./config.yaml')
    base_path = os.path.join(
        args['train']['save_dir'],
        args['train']['model_name'],
        args['train']['task_name']
    )
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Please enable GPU to run this script.")
    
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Initialize model
    model = CLCC(64, 3, 3).to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)
    
    # Initialize training components
    scaler = GradScaler(enabled=args['train']['use_amp'])
    logger = Logger(os.path.join(base_path, 'tensorboard'))
    optimizer, scheduler = configuration_optimizer(model, args)
    
    # Save configuration and print model info
    save_dict_as_yaml(args, base_path)
    print_params_and_macs(model)
    
    # Stage-wise training loop
    num_stages = len(args['train']['stage_epochs'])
    
    for stage_idx in range(num_stages):
        print('\033[92m' + '=' * 60)
        print(f'Start Stage {stage_idx + 1}/{num_stages}')
        print('=' * 60 + '\033[0m')
        
        # Enable resume for stages after the first
        if stage_idx > 0:
            args['train']['resume'] = True
        
        # Configure dataloaders for current stage
        # Use configuration_dataloader for full augmentation
        # Use configuration_dataloader_no_aug for validation-only transforms
        train_loader, valid_loader = configuration_dataloader(args, stage_idx)
        
        print(f"Stage {stage_idx + 1} Info:")
        print(f"  - Epochs: {args['train']['stage_epochs'][stage_idx]}")
        print(f"  - Train batch size: {args['data']['train_batch_size'][stage_idx]}")
        print(f"  - Train image size: {args['data']['train_img_size'][stage_idx]}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Valid batches: {len(valid_loader)}")
        
        # Train the current stage
        train(
            hparams=args,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            logger=logger,
            train_loader=train_loader,
            valid_loader=valid_loader,
            stage_index=stage_idx
        )
        
        print('\033[92m' + '=' * 60)
        print(f'End Stage {stage_idx + 1}/{num_stages}')
        print('=' * 60 + '\033[0m\n')
    
    print('\033[92m' + '=' * 60)
    print('Training Complete!')
    print('=' * 60 + '\033[0m')


if __name__ == '__main__':
    main()