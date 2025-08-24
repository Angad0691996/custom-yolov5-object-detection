# 🛠 Custom YOLOv5 Object Detection

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Custom--Trained-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-Active-success)

A custom-trained **YOLOv5** model detecting 5 unique objects using a self-curated dataset. All images were manually captured and labeled.

---

## 📌 Classes

The model is trained to detect the following 5 objects:

- **company_id**
- **f14_toy_plane**
- **jbl_charging_cable**
- **jbl_headset**
- **wallet**

---

## 📂 Project Structure
custom-yolov5-object-detection/
├── data.yaml # Dataset configuration (paths, class names)
├── scripts/ # Helper scripts (e.g., stats or preprocessing)
├── examples/ # Demo videos and sample images
├── train_result.txt # Training logs (summary)
├── run_command.txt # Inference command for quick reference
├── runs/ # YOLO training outputs (ignored in Git)
├── images/ # Dataset images (ignored in Git)
├── labels/ # YOLO labels (ignored in Git)
└── best.pt # Trained YOLOv5 model weights


---

## ⚡ Training

Run this command to train the model:
yolo task=detect mode=train data=data.yaml model=yolov5s.pt epochs=100 imgsz=640 device=0

🔍 Inference

## Run real-time inference via webcam:
cd custom_yolo_dataset_relabel
yolo task=detect mode=predict model=best.pt source=0 device=0 show=True

##Run inference on an image or video:
yolo task=detect mode=predict model=best.pt source=examples/demo.jpg

## 🎥 Demo Video  
Watch a demo of the model in action here:  
[▶ Watch Demo (Google Drive)](https://drive.google.com/file/d/1bJM3-7dt9BBpAfIOxFMdAFDF4I9GwmbD/view?usp=drive_link)


##📊 Results
Successfully trained on 5 custom classes
Real-time detection (30+ FPS on GPU + CUDA)
Robust performance across varying lighting and angles

##👤 Author
Angad Singh
