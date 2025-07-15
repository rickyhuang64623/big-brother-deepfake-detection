
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml

from models.steganography import SteganoNetwork
from models.template import LearnableTemplate
from models.detector import Detector


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_models(config, checkpoint_path):
    # Initialize models
    stego_net = SteganoNetwork(
        in_channels=config['model']['stego']['in_channels'],
        num_blocks=config['model']['stego']['num_blocks']
    )
    template = LearnableTemplate(
        size=config['model']['template']['output_size']
    )
    detector = Detector(
        in_channels=config['model']['detector']['in_channels']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    stego_net.load_state_dict(checkpoint['stego_net'])
    template.load_state_dict(checkpoint['learnable_template'])
    detector.load_state_dict(checkpoint['detector'])
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stego_net = stego_net.to(device)
    template = template.to(device)
    detector = detector.to(device)
    
    return stego_net, template, detector, device

def process_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

def protect_image(image_path, output_path, stego_net, template, transform, device):
    """Protect an image by embedding the hidden face template."""
    # Load and process image
    image = process_image(image_path, transform)
    image = image.to(device)
    
    with torch.no_grad():
        # Get template
        learnable_template = template(1)
        
        # Embed template into image
        protected_image, _ = stego_net(image, learnable_template)
        
        # Save protected image
        save_image(protected_image, output_path, normalize=True)
        print(f"Protected image saved to: {output_path}")

def detect_manipulation(image_path, stego_net, template, detector, transform, device, output_dir):
    """Detect if a protected image has been manipulated."""
    # Load and process image
    image = process_image(image_path, transform)
    image = image.to(device)
    
    with torch.no_grad():
        # Get template
        learnable_template = template(1)
        
        # Extract template from image
        _, extracted_template = stego_net.reverse(image, learnable_template)

        # Debug: print stats for extracted_template
        print(f"[DEBUG] Extracted template stats - min: {extracted_template.min().item():.4f}, max: {extracted_template.max().item():.4f}, mean: {extracted_template.mean().item():.4f}, has_nan: {torch.isnan(extracted_template).any().item()}")
        
        # Prepare detection input
        detection_input = extracted_template
        
        # Get detection result
        pred, _ = detector(detection_input)
        # Debug: print stats for detector output
        print(f"[DEBUG] Detector output - pred: {pred}, has_nan: {torch.isnan(pred).any().item()}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save extracted template
        save_image(
            extracted_template,
            os.path.join(output_dir, f'{base_name}_extracted_template.png'),
            normalize=True
        )
        
        # Print detection result
        prob = pred.item()
        print(f'\nManipulation detection result:')
        print(f'Probability of being manipulated: {prob*100:.2f}%')
        
        return prob

def main():
    parser = argparse.ArgumentParser(description='Protect images and detect manipulations')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, required=True, choices=['protect', 'detect'],
                       help='Operation mode: protect or detect')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for protected image or directory for detection results')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load models
    stego_net, template, detector, device = load_models(config, args.checkpoint)
    stego_net.eval()
    template.eval()
    detector.eval()
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if args.mode == 'protect':
        protect_image(args.input, args.output, stego_net, template, transform, device)
    else:  # detect
        prob = detect_manipulation(args.input, stego_net, template, detector, transform, device, args.output)

if __name__ == '__main__':
    main() 