import os
import random
import shutil

# Define the paths for your new, relabeled dataset
base_dir = "/home/angad/YOLO_tut/custom_yolo_dataset_relabel"
train_images_dir = os.path.join(base_dir, "images/train")
train_labels_dir = os.path.join(base_dir, "labels/train")
val_images_dir = os.path.join(base_dir, "images/val")
val_labels_dir = os.path.join(base_dir, "labels/val")

# Define the split ratio
split_ratio = 0.20  # 20% of the data will be moved to the validation set

# Get a list of all training image filenames
all_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
num_val_images = int(len(all_images) * split_ratio)

# Randomly select images for validation
val_images = random.sample(all_images, num_val_images)

# Ensure validation directories exist and are clean
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
print("Validation directories are ready.")

# Move the selected images and their labels
print(f"Moving {len(val_images)} images and their labels to the validation set...")
for img_name in val_images:
    label_name = img_name.replace('.jpg', '.txt')
    
    # Define source and destination paths
    src_image = os.path.join(train_images_dir, img_name)
    src_label = os.path.join(train_labels_dir, label_name)
    dst_image = os.path.join(val_images_dir, img_name)
    dst_label = os.path.join(val_labels_dir, label_name)

    # Move the image
    if os.path.exists(src_image):
        shutil.move(src_image, dst_image)
    
    # Move the label
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"Warning: Label file {src_label} not found for image {img_name}. Skipping...")

print("Dataset split is complete.")
print(f"Number of training images remaining: {len(os.listdir(train_images_dir))}")
print(f"Number of validation images: {len(os.listdir(val_images_dir))}")
