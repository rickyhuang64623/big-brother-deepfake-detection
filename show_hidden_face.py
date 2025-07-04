import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from models.template import LearnableTemplate
from models.steganography import SteganoNetwork

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models
template_model = LearnableTemplate(size=128).to(device)
stego_net = SteganoNetwork(in_channels=3, num_blocks=4).to(device)

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_100.pth', map_location=device)
template_model.load_state_dict(checkpoint['learnable_template'])
stego_net.load_state_dict(checkpoint['stego_net'])

# Set models to eval mode
template_model.eval()
stego_net.eval()

# Convert tensors to images
def tensor_to_image(tensor):
    # Convert from [-1, 1] to [0, 1] range
    img = (tensor.squeeze().cpu().detach().numpy() + 1) / 2
    img = img.transpose(1, 2, 0)
    return img

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the test image
real_image = Image.open('test_images/elon.jpg').convert('RGB')
real_image_tensor = transform(real_image).unsqueeze(0).to(device)

with torch.no_grad():
    # Generate template
    template = template_model(1)
    
    # Embed template in real image
    embedded_image = stego_net(real_image_tensor, template)
    
    # Extract template from embedded image
    extracted_template = stego_net(embedded_image, extract_mode=True)

# Create visualization
plt.figure(figsize=(20, 5))

# Show original image
plt.subplot(141)
plt.imshow(tensor_to_image(real_image_tensor))
plt.title('Original Image')
plt.axis('off')

# Show template
plt.subplot(142)
plt.imshow(tensor_to_image(template))
plt.title('Hidden Face Template')
plt.axis('off')

# Show embedded image
plt.subplot(143)
plt.imshow(tensor_to_image(embedded_image))
plt.title('Image with Hidden Face')
plt.axis('off')

# Show extracted template
plt.subplot(144)
plt.imshow(tensor_to_image(extracted_template))
plt.title('Extracted Hidden Face')
plt.axis('off')

plt.tight_layout()
plt.savefig('hidden_face_demonstration.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'hidden_face_demonstration.png'") 