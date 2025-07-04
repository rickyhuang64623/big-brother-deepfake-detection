import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.template import LearnableTemplate

def test_random_template_initialization():
    """Test the random noise template initialization."""
    print("Testing random noise template initialization...")
    
    # Create LearnableTemplate with random initialization
    template = LearnableTemplate(size=64, channels=3)
    
    # Get a batch of templates
    batch_size = 4
    template_batch = template(batch_size)
    
    print(f"Template batch shape: {template_batch.shape}")
    print(f"Expected shape: ({batch_size}, 3, 64, 64)")
    
    # Check shape
    expected_shape = (batch_size, 3, 64, 64)
    assert template_batch.shape == expected_shape, f"Shape mismatch: {template_batch.shape} vs {expected_shape}"
    
    # Check value range (should be in [-1, 1] due to tanh)
    min_val = template_batch.min().item()
    max_val = template_batch.max().item()
    print(f"Value range: [{min_val:.4f}, {max_val:.4f}]")
    
    assert min_val >= -1.0, f"Minimum value {min_val} should be >= -1"
    assert max_val <= 1.0, f"Maximum value {max_val} should be <= 1"
    
    # Check that templates are different (random initialization)
    template1 = template(1)
    template2 = template(1)
    
    # They should be the same since it's the same learnable parameter
    assert torch.allclose(template1, template2), "Templates should be identical for same batch"
    
    print("✓ Random template initialization test passed!")

def test_template_learning():
    """Test that the template can be updated through gradients."""
    print("\nTesting template learning capability...")
    
    # Create template
    template = LearnableTemplate(size=32, channels=3)
    
    # Get initial template
    initial_template = template(1).clone().detach()
    
    # Create a simple loss function
    target = torch.randn_like(initial_template) * 0.5
    loss_fn = nn.MSELoss()
    
    # Calculate loss
    template_batch = template(1)
    loss = loss_fn(template_batch, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients were computed
    assert template.template.grad is not None, "Template gradients should be computed"
    
    print(f"Initial loss: {loss.item():.6f}")
    print(f"Gradient norm: {template.template.grad.norm().item():.6f}")
    
    # Simulate one optimization step
    with torch.no_grad():
        template.template -= 0.01 * template.template.grad
    
    # Get updated template
    updated_template = template(1).clone().detach()
    
    # Check that template changed
    template_change = torch.norm(updated_template - initial_template).item()
    print(f"Template change after optimization: {template_change:.6f}")
    
    assert template_change > 0, "Template should change after optimization"
    
    print("✓ Template learning test passed!")

def test_template_consistency():
    """Test that the template is consistent across calls."""
    print("\nTesting template consistency...")
    
    # Create template
    template = LearnableTemplate(size=64, channels=3)
    
    # Get template multiple times
    template1 = template(1)
    template2 = template(2)
    template3 = template(1)
    
    # Check that same batch size gives same result
    assert torch.allclose(template1, template3), "Same batch size should give same template"
    
    # Check that different batch sizes give same template values (just expanded)
    assert torch.allclose(template1, template2[:1]), "Different batch sizes should give same template values"
    
    print("✓ Template consistency test passed!")

if __name__ == "__main__":
    print("Running random template tests...\n")
    
    try:
        test_random_template_initialization()
        test_template_learning()
        test_template_consistency()
        
        print("\n🎉 All random template tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 