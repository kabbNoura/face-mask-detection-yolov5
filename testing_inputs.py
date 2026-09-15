import subprocess
import sys
import torch
import pathlib
import cv2
import os

"""
sys.path.insert(0, r'./yolov5-master')  # adjust if your folder name differs
pathlib.PosixPath = pathlib.WindowsPath
ckpt = torch.load('best.pt', map_location='cpu', weights_only=False)
torch.save(ckpt, 'best_windows.pt')

subprocess.run([
    "python", "./yolov5-master/detect.py",
    "--weights", "best_windows.pt",
    "--source", "./testing/input4.jpg",
    "--conf", "0.40"
])
"""

# fix for Colab-trained weights on Windows
pathlib.PosixPath = pathlib.WindowsPath

WEIGHTS = "best_windows.pt"
YOLO_DIR = "yolov5-master"

FOLDER = "testing"
FILES = [
    "input3.jpg", "output3.jpg",
    "input4.jpg", "output4.jpg",
    "input1.mp4", "output1.mp4",
    "input2.mp4", "output2.mp4",
]

WINDOW = "Viewer"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)


def show_image(path):
    """Returns 'next' or 'prev' depending on key pressed."""
    img = cv2.imread(path)
    while True:
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # space -> next
            return "next"
        if key == 8:  # backspace -> prev
            return "prev"
        if key == 27:
            cv2.destroyAllWindows()
            exit()


def show_video(path):
    """Returns 'next' or 'prev' depending on key pressed."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == 32:
            cap.release()
            return "next"
        if key == 8:
            cap.release()
            return "prev"
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
    cap.release()

    # Video finished naturally: freeze on last frame until a key is pressed
    if last_frame is not None:
        while True:
            cv2.imshow(WINDOW, last_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 32:
                return "next"
            if key == 8:
                return "prev"
            if key == 27:
                cv2.destroyAllWindows()
                exit()


def run_webcam():
    print("Loading model for live webcam... (ESC to quit)")
    model = torch.hub.load(YOLO_DIR, "custom", path=WEIGHTS, source="local")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb)
        annotated = cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()


i = 0
while 0 <= i < len(FILES):
    fname = FILES[i]
    path = os.path.join(FOLDER, fname)
    print(f"Showing: {fname}")
    if fname.lower().endswith((".mp4", ".avi", ".mov")):
        action = show_video(path)
    else:
        action = show_image(path)

    i += 1 if action == "next" else -1
    i = max(i, 0)  # don't go before the first file

run_webcam()
cv2.destroyAllWindows()
print("Done.")
