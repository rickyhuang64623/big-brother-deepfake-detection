import os
import argparse
from pathlib import Path

def create_directory_structure(base_path):
    """Create the required directory structure for the dataset."""
    # Create main directories
    directories = [
        'data/train/real',  # Only real images needed for training
        'data/val/real',    # Only real images needed for validation
        'checkpoints',
        'logs',
        'outputs'
    ]
    
    for directory in directories:
        path = Path(base_path) / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f'Created directory: {path}')

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset directory structure')
    parser.add_argument('--base_path', type=str, default='.',
                       help='Base path where directories will be created')
    
    args = parser.parse_args()
    create_directory_structure(args.base_path)
    
    print('\nDirectory structure created successfully!')
    print('\nPlease organize your dataset as follows:')
    print('- Place real face images in data/train/real (for training)')
    print('- Place a few different real face images in data/val/real (for validation)')
    print('\nRecommended amounts:')
    print('- Training: At least 100 images')
    print('- Validation: 20-30 images')
    print('\nSupported image formats: .jpg, .png')
    print('\nThe system will automatically generate manipulated versions for training.')

if __name__ == '__main__':
    main() 