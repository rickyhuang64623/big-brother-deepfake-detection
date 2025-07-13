import os, shutil, random

src_dir = "img_align_celeba"  # your image folder
train_dir = "train"
test_dir = "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
random.shuffle(files)
split = int(0.8 * len(files))

for f in files[:split]:
    shutil.copy(os.path.join(src_dir, f), os.path.join(train_dir, f))
for f in files[split:]:
    shutil.copy(os.path.join(src_dir, f), os.path.join(test_dir, f))
