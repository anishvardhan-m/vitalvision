import sys

import cv2
import mediapipe as mp
import numpy
import scipy


def main() -> None:
    print(f"Python:    {sys.version.split()[0]}")
    print(f"OpenCV:    {cv2.__version__}")
    print(f"MediaPipe: {mp.__version__}")
    print(f"NumPy:     {numpy.__version__}")
    print(f"SciPy:     {scipy.__version__}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        sys.exit(1)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("ERROR: could not read a frame from webcam")
        sys.exit(1)

    print(f"Webcam frame shape: {frame.shape}")


if __name__ == "__main__":
    main()
