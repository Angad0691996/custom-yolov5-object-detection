import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Set the source to your webcam (usually 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to exit.")

# Loop through the video frames
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame from webcam.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
