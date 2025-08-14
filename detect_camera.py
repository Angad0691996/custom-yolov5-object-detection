import cv2

def list_cameras():
    i = 0
    while True:
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        print(f"Camera found at index: {i}")
        cap.release()
        i += 1

list_cameras()
