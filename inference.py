import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import argparse
import yaml
from types import SimpleNamespace
import os

from models.steganography import SteganoNetwork
from models.template import LearnableTemplate

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def load_image(image_path, model_size=512):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Store original size
    original_size = image.size
    
    # Calculate aspect ratio preserving size
    aspect_ratio = original_size[0] / original_size[1]
    if aspect_ratio > 1:
        new_size = (int(model_size * aspect_ratio), model_size)
    else:
        new_size = (model_size, int(model_size / aspect_ratio))
    
    # Create transform for model input
    transform = transforms.Compose([
        transforms.Resize(new_size, Image.Resampling.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transform image for model
    tensor = transform(image).unsqueeze(0)
    
    return tensor, original_size, new_size

def enhance_image(image):
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)  # Slight sharpness enhancement
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)  # Slight contrast enhancement
    
    return image

def save_image(tensor, path, original_size, new_size):
    # Denormalize
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    image = transforms.ToPILImage()(tensor)
    
    # Apply enhancements
    image = enhance_image(image)
    
    # Resize back to original size using high-quality interpolation
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    
    # Save with high quality
    image.save(path, 'JPEG', quality=95, subsampling=0)

def process_in_patches(image_tensor, template_tensor, stego_net, patch_size=128, overlap=32):
    """Process large images in patches to maintain quality."""
    B, C, H, W = image_tensor.size()
    
    # Pad the input image if needed
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    # Initialize output tensor
    output = torch.zeros_like(image_tensor)
    count = torch.zeros_like(image_tensor)
    
    # Process each patch
    for y in range(0, H + pad_h - patch_size + 1, patch_size - overlap):
        for x in range(0, W + pad_w - patch_size + 1, patch_size - overlap):
            # Extract patch
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # Resize template to match patch size
            template_patch = F.interpolate(template_tensor, size=(patch_size, patch_size), mode='bilinear', align_corners=True)
            
            # Process patch
            with torch.no_grad():
                processed_patch = stego_net(patch, template_patch)
            
            # Create blending weights
            weights = torch.ones_like(patch)
            if overlap > 0:
                # Fade out edges in overlap regions
                for i in range(overlap):
                    # Fade left edge
                    if x > 0:
                        weights[:, :, :, i] *= i / overlap
                    # Fade right edge
                    if x < W - patch_size:
                        weights[:, :, :, -(i+1)] *= i / overlap
                    # Fade top edge
                    if y > 0:
                        weights[:, :, i, :] *= i / overlap
                    # Fade bottom edge
                    if y < H - patch_size:
                        weights[:, :, -(i+1), :] *= i / overlap
            
            # Add processed patch to output
            output[:, :, y:y+patch_size, x:x+patch_size] += processed_patch * weights
            count[:, :, y:y+patch_size, x:x+patch_size] += weights
    
    # Average overlapping regions
    output = output / (count + 1e-6)
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        output = output[:, :, :H, :W]
    
    return output

def main():
    parser = argparse.ArgumentParser(description='Add hidden face to image')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_100.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--model_size', type=int, default=512,
                       help='Size to process image at (default: 512)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    stego_net = SteganoNetwork(
        in_channels=3,
        num_blocks=4
    ).to(device)
    
    learnable_template = LearnableTemplate(
        size=128,  # Keep original template size
        channels=3
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stego_net.load_state_dict(checkpoint['stego_net'])
    learnable_template.load_state_dict(checkpoint['learnable_template'])

    # Set models to eval mode
    stego_net.eval()
    learnable_template.eval()

    # Load input image
    input_image, original_size, new_size = load_image(args.input, args.model_size)
    input_image = input_image.to(device)
    
    print(f"Processing at size: {new_size[0]}x{new_size[1]}")
    
    # Generate template
    with torch.no_grad():
        template = learnable_template(batch_size=1)
        
        # Process image in patches
        output_image = process_in_patches(input_image, template, stego_net)
    
    # Save original size version
    save_image(input_image, 'original_' + os.path.basename(args.output), original_size, new_size)
    
    # Save result
    save_image(output_image, args.output, original_size, new_size)
    print(f"Hidden face added successfully!")
    print(f"Original image saved as: original_{os.path.basename(args.output)}")
    print(f"Result saved as: {args.output}")
    print(f"Original resolution maintained: {original_size[0]}x{original_size[1]}")

if __name__ == '__main__':
    main() 