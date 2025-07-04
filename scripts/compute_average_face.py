import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    img = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    return img

def main():
    train_dir = 'data/train/real'  # Adjust if your real faces are elsewhere
    output_size = 128  # Match your template size
    image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_paths:
        print('No images found in', train_dir)
        return
    
    sum_img = None
    count = 0
    for path in tqdm(image_paths, desc='Averaging faces'):
        img = load_image(path, size=output_size)
        if sum_img is None:
            sum_img = np.zeros_like(img)
        sum_img += img
        count += 1
    
    avg_img = sum_img / count
    avg_img_tensor = torch.from_numpy(avg_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # Save as PNG
    avg_img_uint8 = (np.clip(avg_img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(avg_img_uint8).save('average_face.png')
    print('Saved average face as average_face.png')
    
    # Save as tensor
    torch.save(avg_img_tensor, 'average_face.pt')
    print('Saved average face tensor as average_face.pt')

if __name__ == '__main__':
    main() 