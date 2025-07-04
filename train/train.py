import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
import time
import random
import cv2
import numpy as np
import wandb
import matplotlib.pyplot as plt

from data.dataset import create_dataloaders
from models.steganography import SteganoNetwork, SteganoLoss
from models.template import TemplateGenerator, TemplateLoss, LearnableTemplate
from models.detector import Detector, DetectorLoss

def apply_benign_manipulations(images, p=0.5):
    """Apply random benign manipulations to images."""
    batch_size = images.size(0)
    manipulated = images.clone()
    
    for i in range(batch_size):
        if random.random() < p:
            # Convert to numpy for OpenCV operations
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            img = (img * 255).astype(np.uint8)
            
            # Apply random benign manipulations
            if random.random() < 0.3:
                # JPEG compression
                quality = random.randint(60, 95)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', img, encode_param)
                img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            if random.random() < 0.3:
                # Gaussian blur
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.5, 1.5)
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            
            if random.random() < 0.2:
                # Resize and back
                scale = random.uniform(0.8, 1.2)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))
                img = cv2.resize(img, (w, h))
            
            if random.random() < 0.2:
                # Add light noise
                noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.uint8)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Convert back to tensor
            img = img.astype(np.float32) / 255.0
            img = img * 2 - 1  # Convert from [0, 1] to [-1, 1]
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            manipulated[i] = img
    
    return manipulated.to(images.device)

def apply_malicious_forgery(images, p=0.5):
    """Apply Self-Blended Images (SBI) method for malicious forgery simulation."""
    batch_size = images.size(0)
    forged = images.clone()
    
    for i in range(batch_size):
        if random.random() < p:
            # Convert to numpy for OpenCV operations
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            img = (img * 255).astype(np.uint8)
            
            # SBI method: Create a mask and blend with a shifted version
            h, w = img.shape[:2]
            
            # Create random mask (simulating face swap region)
            mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            radius_x = random.randint(w // 6, w // 3)
            radius_y = random.randint(h // 6, h // 3)
            
            # Create elliptical mask
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 
                       0, 0, 360, 255, -1)
            
            # Apply Gaussian blur to mask edges
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = mask.astype(np.float32) / 255.0
            
            # Create shifted version (simulating different face)
            shift_x = random.randint(-w // 8, w // 8)
            shift_y = random.randint(-h // 8, h // 8)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted = cv2.warpAffine(img, M, (w, h))
            
            # Blend images using the mask
            mask_3d = np.stack([mask] * 3, axis=2)
            blended = img * (1 - mask_3d) + shifted * mask_3d
            
            # Convert back to tensor
            blended = blended.astype(np.float32) / 255.0
            blended = blended * 2 - 1  # Convert from [0, 1] to [-1, 1]
            blended = torch.from_numpy(blended.transpose(2, 0, 1)).float()
            forged[i] = blended
    
    return forged.to(images.device)

def dict_to_namespace(d):
    """Convert dictionary to namespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return dict_to_namespace(config)

def main():
    import torch
    # torch.autograd.set_detect_anomaly(True)  # Temporarily disabled
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize Weights & Biases
    wandb.init(
        project="steganography-deepfake-detection",
        name=f"inn-stego-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "num_epochs": config.training.num_epochs,
            "batch_size": config.training.batch_size,
            "learning_rate_stego": config.training.learning_rate.stego,
            "learning_rate_template": config.training.learning_rate.template,
            "learning_rate_detector": config.training.learning_rate.detector,
            "stego_blocks": config.model.stego.num_blocks,
            "template_latent_dim": config.model.template.latent_dim,
            "lambda_cover": config.loss.stego.lambda_cover,
            "lambda_secret": config.loss.stego.lambda_secret,
            "lambda_remains": config.loss.stego.lambda_remains,
            "lambda_sec": config.loss.stego.lambda_sec,
        }
    )
    
    # Create directories
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.log_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config.device)
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(config)
    print(f'Training with {len(train_loader.dataset)} images')
    print(f'Validating with {len(val_loader.dataset)} images')
    
    # Initialize models
    stego_net = SteganoNetwork(
        in_channels=config.model.stego.in_channels,
        num_blocks=config.model.stego.num_blocks
    ).to(device)
    
    template_gen = TemplateGenerator(
        latent_dim=config.model.template.latent_dim,
        output_size=config.model.template.output_size
    ).to(device)
    
    learnable_template = LearnableTemplate(
        size=config.model.template.output_size
    ).to(device)
    
    detector = Detector(
        in_channels=config.model.detector.in_channels
    ).to(device)
    
    # Initialize loss functions
    stego_loss_fn = SteganoLoss(
        lambda_cover=config.loss.stego.lambda_cover,
        lambda_secret=config.loss.stego.lambda_secret,
        lambda_remains=config.loss.stego.lambda_remains
    ).to(device)
    template_loss_fn = TemplateLoss(lambda_reg=config.loss.template.lambda_reg, avg_face_path=os.path.join(os.path.dirname(__file__), '..', 'average_face.pt')).to(device)
    detector_loss_fn = DetectorLoss(lambda_integrity=1.0, lambda_template=0.1).to(device)
    
    # Initialize optimizers
    stego_optimizer = torch.optim.Adam(
        stego_net.parameters(),
        lr=config.training.learning_rate.stego
    )
    
    template_optimizer = torch.optim.Adam([
        {'params': template_gen.parameters(), 'lr': config.training.learning_rate.template},
        {'params': learnable_template.parameters(), 'lr': config.training.learning_rate.template}
    ])
    
    detector_optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=config.training.learning_rate.detector
    )
    
    # Setup tensorboard
    writer = SummaryWriter(config.paths.log_dir)
    
    # Training loop
    print('Starting training...')
    start_time = time.time()
    
    for epoch in range(config.training.num_epochs):
        stego_net.train()
        template_gen.train()
        learnable_template.train()
        detector.train()
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs}')
        
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            batch_start_time = time.time()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Generate template
            z = torch.randn(batch_size, config.model.template.latent_dim).to(device)
            template = template_gen(z)
            learnable_template_batch = learnable_template(batch_size)
            
            # Step 1: Protect authentic images with steganography
            # X = Stego(I, Learnable Template S)
            protected_images, stego_template_remains = stego_net(real_images, learnable_template_batch)
            
            # Step 2: Generate real samples with benign manipulations g+(X)
            # Apply random benign manipulations (JPEG compression, blurring, resizing, light noise)
            real_manipulated = apply_benign_manipulations(protected_images)
            
            # Step 3: Generate fake samples with malicious forgery g-(X)
            # Apply Self-Blended Images (SBI) method for malicious forgery
            fake_manipulated = apply_malicious_forgery(protected_images)
            
            # Train steganography network
            stego_optimizer.zero_grad()
            
            # Reverse the process to get the extracted template
            _, extracted_template = stego_net.reverse(protected_images.detach(), stego_template_remains.detach())
            
            stego_total_loss, stego_loss_cover, stego_loss_secret, stego_loss_remains = stego_loss_fn(
                real_images, protected_images, learnable_template_batch, extracted_template, stego_template_remains
            )
            
            # Add L_sec loss (equation 5) from the INN
            l_sec_loss = stego_net.inn.calculate_l_sec_loss(real_images, learnable_template_batch)
            
            # Combine all steganography losses with appropriate weights
            lambda_sec = config.loss.stego.lambda_sec  # Weight for L_sec loss (from config)
            stego_total_loss_with_sec = stego_total_loss + lambda_sec * l_sec_loss
            
            stego_total_loss_with_sec.backward(retain_graph=True)
            stego_optimizer.step()
            
            # Train template generator and learnable template
            template_optimizer.zero_grad()
            template_total_loss, template_reg_loss, template_sim_loss = template_loss_fn(learnable_template_batch)
            template_total_loss.backward(retain_graph=True)
            
            # Monitor template gradients
            if batch_idx == 0:  # Only print every epoch
                template_grad_norm = learnable_template.template.grad.norm().item() if learnable_template.template.grad is not None else 0.0
                print(f"Epoch {epoch+1}: Template gradient norm: {template_grad_norm:.6f}")
            
            template_optimizer.step()
            
            # Train detector
            detector_optimizer.zero_grad()
            
            # Real samples (benignly manipulated) - detach to prevent gradient issues
            real_integrity, real_extracted_template = detector(real_manipulated.detach())
            real_labels = torch.ones(batch_size, 1).to(device)
            
            # Fake samples (maliciously forged) - detach to prevent gradient issues
            fake_integrity, fake_extracted_template = detector(fake_manipulated.detach())
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Detection loss - compare extracted templates to the original template
            detection_loss, integrity_loss, template_loss = detector_loss_fn(
                torch.cat([real_integrity, fake_integrity]),
                torch.cat([real_extracted_template, fake_extracted_template]),
                torch.cat([real_labels, fake_labels]),
                torch.cat([learnable_template_batch, learnable_template_batch])  # Target template
            )
            detection_loss.backward()
            detector_optimizer.step()
            
            # Calculate total loss
            total_loss = stego_total_loss_with_sec.item() + template_total_loss.item() + detection_loss.item()
            epoch_loss += total_loss
            
            # Calculate time per batch
            batch_time = time.time() - batch_start_time
            
            # Update progress bar with detailed metrics
            progress_bar.set_postfix({
                'loss': f'{total_loss:.4f}',
                's_cover': f'{stego_loss_cover.item():.4f}',
                's_secret': f'{stego_loss_secret.item():.4f}',
                's_remains': f'{stego_loss_remains.item():.4f}',
                's_lsec': f'{l_sec_loss.item():.4f}',
                'd_int': f'{integrity_loss.item():.4f}',
                'd_tmpl': f'{template_loss.item():.4f}',
                'batch_time': f'{batch_time:.2f}s'
            })
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/total', total_loss, global_step)
            writer.add_scalar('Loss/stego_total', stego_total_loss_with_sec.item(), global_step)
            writer.add_scalar('Loss/stego_cover', stego_loss_cover.item(), global_step)
            writer.add_scalar('Loss/stego_secret', stego_loss_secret.item(), global_step)
            writer.add_scalar('Loss/stego_remains', stego_loss_remains.item(), global_step)
            writer.add_scalar('Loss/stego_lsec', l_sec_loss.item(), global_step)
            writer.add_scalar('Loss/template', template_total_loss.item(), global_step)
            writer.add_scalar('Loss/detection', detection_loss.item(), global_step)
            writer.add_scalar('Loss/detection_integrity', integrity_loss.item(), global_step)
            writer.add_scalar('Loss/detection_template', template_loss.item(), global_step)
            writer.add_scalar('Time/batch', batch_time, global_step)
            
            # Log to Weights & Biases
            wandb.log({
                "loss/total": total_loss,
                "loss/stego_total": stego_total_loss_with_sec.item(),
                "loss/stego_cover": stego_loss_cover.item(),
                "loss/stego_secret": stego_loss_secret.item(),
                "loss/stego_remains": stego_loss_remains.item(),
                "loss/stego_lsec": l_sec_loss.item(),
                "loss/template": template_total_loss.item(),
                "loss/detection": detection_loss.item(),
                "loss/detection_integrity": integrity_loss.item(),
                "loss/detection_template": template_loss.item(),
                "time/batch": batch_time,
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
            }, step=global_step)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Time Elapsed: {epoch_time/60:.2f} minutes')

        # --- Save and visualize the hidden template ---
        template_tensor = learnable_template(1).detach().cpu().squeeze(0)  # (3, H, W)
        template_np = template_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        template_np = (template_np + 1) / 2  # [-1,1] -> [0,1]
        template_np = np.clip(template_np, 0, 1)
        os.makedirs(config.paths.output_dir, exist_ok=True)
        template_path = os.path.join(config.paths.output_dir, f'template_epoch_{epoch+1}.png')
        plt.imsave(template_path, template_np)
        print(f'Hidden template saved to {template_path}')

        # Log epoch summary to wandb
        wandb.log({
            "epoch/avg_loss": avg_loss,
            "epoch/time_minutes": epoch_time/60,
            "epoch/epoch": epoch + 1,
            f"template/epoch_{epoch+1}": wandb.Image(template_path),
        })
        
        # Save checkpoint
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'stego_net': stego_net.state_dict(),
                'template_gen': template_gen.state_dict(),
                'learnable_template': learnable_template.state_dict(),
                'detector': detector.state_dict(),
                'stego_optimizer': stego_optimizer.state_dict(),
                'template_optimizer': template_optimizer.state_dict(),
                'detector_optimizer': detector_optimizer.state_dict()
            }
            
            save_path = os.path.join(config.paths.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            print(f'Checkpoint saved to {save_path}')
            
            # Log model artifact to wandb
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch+1}",
                type="model",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
    
    # Finish wandb run
    wandb.finish()
    print("Training completed!")

if __name__ == '__main__':
    main() 