import os
import random
import shutil

# Base directory
base_dir = os.getcwd()
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

# Create output directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

# 80-20 split
split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move function
def move_pairs(img_list, split):
    for img in img_list:
        label_file = img.rsplit('.', 1)[0] + '.txt'
        src_img = os.path.join(image_dir, img)
        dst_img = os.path.join(image_dir, split, img)

        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(label_dir, split, label_file)

        # Move image
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        # Move label
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

move_pairs(train_images, 'train')
move_pairs(val_images, 'val')

print("✅ Dataset split complete. You can now re-run training.")
