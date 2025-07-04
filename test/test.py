import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml

from models.steganography import InvertibleSteganography
from models.template import LearnableTemplate
from models.detector import DeepfakeDetector

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_models(config, checkpoint_path):
    # Initialize models
    stego_net = InvertibleSteganography(
        in_channels=config['model']['stego']['in_channels'],
        num_blocks=config['model']['stego']['num_blocks']
    )
    template = LearnableTemplate(
        size=config['model']['template']['output_size']
    )
    detector = DeepfakeDetector(
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

def main():
    parser = argparse.ArgumentParser(description='Test deepfake detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    
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
    
    # Process input image
    image = process_image(args.input, transform)
    image = image.to(device)
    
    with torch.no_grad():
        # Get template
        learnable_template = template(1)
        
        # Extract template from image
        extracted_template = stego_net(image, learnable_template, reverse=True)
        
        # Prepare detection input
        detection_input = torch.cat([learnable_template, extracted_template], dim=1)
        
        # Get detection result
        pred, _ = detector(detection_input)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Save extracted template
        save_image(
            extracted_template,
            os.path.join(args.output_dir, f'{base_name}_extracted_template.png'),
            normalize=True
        )
        
        # Save original template
        save_image(
            learnable_template,
            os.path.join(args.output_dir, f'{base_name}_original_template.png'),
            normalize=True
        )
        
        # Print detection result
        prob = pred.item()
        print(f'\nDeepfake detection result:')
        print(f'Probability of being real: {(1-prob)*100:.2f}%')
        print(f'Probability of being fake: {prob*100:.2f}%')
        
        with open(os.path.join(args.output_dir, f'{base_name}_result.txt'), 'w') as f:
            f.write(f'Deepfake detection result:\n')
            f.write(f'Probability of being real: {(1-prob)*100:.2f}%\n')
            f.write(f'Probability of being fake: {prob*100:.2f}%\n')

if __name__ == '__main__':
    main() 