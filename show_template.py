from PIL import Image
import matplotlib.pyplot as plt

# Load and display the image
img = Image.open('template_visualization.png')
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show() 