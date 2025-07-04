import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inn import InvertibleSteganoNetwork, DWT_LL

def test_dwt_ll():
    """Test the DWT_LL module functionality."""
    print("Testing DWT_LL module...")
    
    # Create DWT_LL module
    dwt_ll = DWT_LL()
    
    # Create test input
    batch_size = 2
    channels = 3
    height, width = 64, 64
    
    # Create random test images
    test_image = torch.randn(batch_size, channels, height, width)
    
    # Apply DWT_LL
    ll_coeffs = dwt_ll(test_image)
    
    print(f"Input shape: {test_image.shape}")
    print(f"LL coefficients shape: {ll_coeffs.shape}")
    print(f"Expected LL shape: ({batch_size}, {channels}, {height//2}, {width//2})")
    
    # Check if output shape is correct
    expected_shape = (batch_size, channels, height//2, width//2)
    assert ll_coeffs.shape == expected_shape, f"Shape mismatch: {ll_coeffs.shape} vs {expected_shape}"
    
    print("✓ DWT_LL test passed!")

def test_l_sec_loss():
    """Test the L_sec loss calculation."""
    print("\nTesting L_sec loss calculation...")
    
    # Create InvertibleSteganoNetwork
    inn = InvertibleSteganoNetwork(num_blocks=4, in_channels=3)
    
    # Create test inputs
    batch_size = 2
    channels = 3
    height, width = 64, 64
    
    # Create random test images and templates
    test_image = torch.randn(batch_size, channels, height, width)
    test_template = torch.randn(batch_size, channels, height, width)
    
    # Calculate L_sec loss
    l_sec_loss = inn.calculate_l_sec_loss(test_image, test_template)
    
    print(f"Image shape: {test_image.shape}")
    print(f"Template shape: {test_template.shape}")
    print(f"L_sec loss value: {l_sec_loss.item():.6f}")
    
    # Check if loss is a scalar tensor
    assert l_sec_loss.dim() == 0, f"Loss should be scalar, got shape {l_sec_loss.shape}"
    assert l_sec_loss.item() >= 0, "Loss should be non-negative"
    
    print("✓ L_sec loss test passed!")

def test_inn_forward_reverse():
    """Test the INN forward and reverse operations."""
    print("\nTesting INN forward and reverse operations...")
    
    # Create InvertibleSteganoNetwork
    inn = InvertibleSteganoNetwork(num_blocks=4, in_channels=3)
    
    # Create test inputs
    batch_size = 2
    channels = 3
    height, width = 64, 64
    
    # Create random test images and templates
    test_image = torch.randn(batch_size, channels, height, width)
    test_template = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    stego_image, stego_template_remains = inn.forward(test_image, test_template)
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Original template shape: {test_template.shape}")
    print(f"Stego image shape: {stego_image.shape}")
    print(f"Stego template remains shape: {stego_template_remains.shape}")
    
    # Reverse pass
    reconstructed_image, reconstructed_template = inn.reverse(stego_image, stego_template_remains)
    
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    print(f"Reconstructed template shape: {reconstructed_template.shape}")
    
    # Check shapes
    assert stego_image.shape == test_image.shape, "Stego image shape mismatch"
    assert stego_template_remains.shape == test_template.shape, "Stego template remains shape mismatch"
    assert reconstructed_image.shape == test_image.shape, "Reconstructed image shape mismatch"
    assert reconstructed_template.shape == test_template.shape, "Reconstructed template shape mismatch"
    
    print("✓ INN forward/reverse test passed!")

def test_loss_gradients():
    """Test that the L_sec loss produces gradients."""
    print("\nTesting L_sec loss gradients...")
    
    # Create InvertibleSteganoNetwork
    inn = InvertibleSteganoNetwork(num_blocks=4, in_channels=3)
    
    # Create test inputs
    batch_size = 2
    channels = 3
    height, width = 64, 64
    
    # Create random test images and templates
    test_image = torch.randn(batch_size, channels, height, width, requires_grad=True)
    test_template = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    # Calculate L_sec loss
    l_sec_loss = inn.calculate_l_sec_loss(test_image, test_template)
    
    # Backward pass
    l_sec_loss.backward()
    
    # Check gradients
    assert test_image.grad is not None, "Image gradients should be computed"
    assert test_template.grad is not None, "Template gradients should be computed"
    
    print(f"Image gradient norm: {test_image.grad.norm().item():.6f}")
    print(f"Template gradient norm: {test_template.grad.norm().item():.6f}")
    
    print("✓ L_sec loss gradients test passed!")

if __name__ == "__main__":
    print("Running L_sec loss tests...\n")
    
    try:
        test_dwt_ll()
        test_l_sec_loss()
        test_inn_forward_reverse()
        test_loss_gradients()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 