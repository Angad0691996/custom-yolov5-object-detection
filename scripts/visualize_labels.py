import cv2
import os

# Define the root path of your dataset
root_path = "/home/angad/YOLO_tut/custom_yolo_dataset_relabel"

# Define the paths to your images and labels
image_dir = os.path.join(root_path, "images/train")
label_dir = os.path.join(root_path, "labels/train")

# Define your class names based on your data.yaml file
class_names = ['company_id', 'f14_toy_plane', 'jbl_charging_cable', 'jbl_headset', 'wallet']

def visualize_labels(image_path, label_path):
    """
    Loads an image and its corresponding YOLO labels, then draws the bounding boxes and resizes the window if needed.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    h, w, _ = image.shape

    # Get screen dimensions
    try:
        import tkinter
        root = tkinter.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.withdraw()
    except ImportError:
        print("tkinter not available, defaulting to 1920x1080.")
        screen_width = 1920
        screen_height = 1080

    # Calculate resize ratio if image is larger than screen
    resize_ratio = 1.0
    if w > screen_width or h > screen_height:
        resize_ratio = min(screen_width / w, screen_height / h) * 0.9 # 90% of screen size
    
    # Resize the image if necessary
    if resize_ratio < 1.0:
        resized_width = int(w * resize_ratio)
        resized_height = int(h * resize_ratio)
        resized_image = cv2.resize(image, (resized_width, resized_height))
    else:
        resized_image = image

    # Read the label file
    if not os.path.exists(label_path):
        print(f"Label file not found for {image_path}")
        # Display the image without labels if no label file exists
        cv2.imshow("Label Visualization", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    with open(label_path, 'r') as f:
        labels = f.readlines()

    # Calculate scaling factor for label coordinates
    scale_x = resized_image.shape[1] / w
    scale_y = resized_image.shape[0] / h

    # Draw bounding boxes on the resized image
    for label in labels:
        parts = label.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, parts)
        class_id = int(class_id)

        # Convert normalized coordinates to pixel coordinates
        x_center_resized = int(x_center * resized_image.shape[1])
        y_center_resized = int(y_center * resized_image.shape[0])
        width_resized = int(width * resized_image.shape[1])
        height_resized = int(height * resized_image.shape[0])

        x1 = int(x_center_resized - width_resized / 2)
        y1 = int(y_center_resized - height_resized / 2)
        x2 = int(x_center_resized + width_resized / 2)
        y2 = int(y_center_resized + height_resized / 2)

        # Define color and thickness
        color = (0, 255, 0)
        thickness = 2

        # Draw rectangle and class name
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(resized_image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display the resized image
    cv2.imshow("Label Visualization", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Get a list of all images
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Visualize each image and its labels
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_file = image_file.replace('.jpg', '.txt')
    label_path = os.path.join(label_dir, label_file)

    visualize_labels(image_path, label_path)

print("Visualization complete. Please check the displayed images.")
