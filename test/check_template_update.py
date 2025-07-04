import torch
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.template import LearnableTemplate

def check_template_initialization():
    """Check if the template is being initialized from the updated average face."""
    print("Checking template initialization...")
    
    # Create template
    template = LearnableTemplate(size=128, channels=3)
    
    # Get template
    template_tensor = template(1)
    
    print(f"Template shape: {template_tensor.shape}")
    print(f"Template value range: [{template_tensor.min().item():.4f}, {template_tensor.max().item():.4f}]")
    print(f"Template mean: {template_tensor.mean().item():.4f}")
    print(f"Template std: {template_tensor.std().item():.4f}")
    
    # Check if average face file exists and its stats
    avg_face_path = os.path.join(os.path.dirname(__file__), '..', 'average_face.pt')
    if os.path.exists(avg_face_path):
        avg_face = torch.load(avg_face_path)
        print(f"\nAverage face shape: {avg_face.shape}")
        print(f"Average face value range: [{avg_face.min().item():.4f}, {avg_face.max().item():.4f}]")
        print(f"Average face mean: {avg_face.mean().item():.4f}")
        print(f"Average face std: {avg_face.std().item():.4f}")
        
        # Check if template matches average face
        if torch.allclose(template_tensor, avg_face, atol=1e-6):
            print("\n✅ Template matches average face exactly")
        else:
            print("\n❌ Template does NOT match average face")
            diff = torch.norm(template_tensor - avg_face).item()
            print(f"Difference norm: {diff:.6f}")
    else:
        print(f"\n❌ Average face file not found at {avg_face_path}")

if __name__ == "__main__":
    check_template_initialization() 