import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.template import LearnableTemplate

def visualize_template():
    """Visualize the learnable template with random initialization."""
    
    # Create template with random initialization
    template = LearnableTemplate(size=128, channels=3)
    
    # Get template
    template_tensor = template(1)  # Shape: (1, 3, 128, 128)
    
    # Convert to numpy for visualization
    template_np = template_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Normalize to [0, 1] for visualization
    template_np = (template_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
    template_np = np.clip(template_np, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB channels
    channel_names = ['Red', 'Green', 'Blue']
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.imshow(template_np[:, :, i], cmap='gray')
        ax.set_title(f'{name} Channel')
        ax.axis('off')
    
    plt.suptitle('Learnable Template - Random Noise Initialization', fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('template_visualization_random.png', dpi=150, bbox_inches='tight')
    print("Template visualization saved as 'template_visualization_random.png'")
    
    # Show the plot
    plt.show()
    
    # Print template statistics
    print(f"\nTemplate Statistics:")
    print(f"Shape: {template_tensor.shape}")
    print(f"Value range: [{template_tensor.min().item():.4f}, {template_tensor.max().item():.4f}]")
    print(f"Mean: {template_tensor.mean().item():.4f}")
    print(f"Std: {template_tensor.std().item():.4f}")

if __name__ == "__main__":
    visualize_template() 