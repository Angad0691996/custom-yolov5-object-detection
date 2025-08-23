import os
import cv2
import glob
import albumentations as A
import yaml

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.Blur(blur_limit=(3, 7), p=0.1),
    A.CLAHE(p=0.1),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.1),
    A.RandomRotate90(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Define paths relative to the script location
original_images_dir = 'images/train'
original_labels_dir = 'labels/train'
augmented_images_dir = 'images/train_aug'
augmented_labels_dir = 'labels/train_aug'

# Create new directories
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# Load list of all original images
image_files = glob.glob(os.path.join(original_images_dir, '*.jpg'))
image_files.extend(glob.glob(os.path.join(original_images_dir, '*.jpeg')))
image_files.extend(glob.glob(os.path.join(original_images_dir, '*.png')))

# Loop through each original image and augment it
for img_path in image_files:
    base_name = os.path.basename(img_path)
    file_name = os.path.splitext(base_name)[0]
    label_path = os.path.join(original_labels_dir, file_name + '.txt')

    if not os.path.exists(label_path):
        print(f"Skipping {img_path}, no corresponding label file found.")
        continue

    # Read the image and labels
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    
    if not labels:
        continue # Skip if no objects are present in the image
    
    bboxes = [[float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
    class_labels = [int(l[0]) for l in labels]
    
    # Generate 10 augmented versions for each original image
    for i in range(1, 11): # Augment 10 times
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            # Save augmented image
            aug_img_name = f'aug_{i}_{file_name}{os.path.splitext(base_name)[1]}'
            aug_img_path = os.path.join(augmented_images_dir, aug_img_name)
            cv2.imwrite(aug_img_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            
            # Save augmented labels
            aug_label_name = f'aug_{i}_{file_name}.txt'
            aug_label_path = os.path.join(augmented_labels_dir, aug_label_name)
            with open(aug_label_path, 'w') as f:
                for bbox, cls_label in zip(transformed['bboxes'], transformed['class_labels']):
                    f.write(f"{int(cls_label)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        except Exception as e:
            print(f"Error during augmentation for {img_path}: {e}")
            
print("Augmentation complete!")
