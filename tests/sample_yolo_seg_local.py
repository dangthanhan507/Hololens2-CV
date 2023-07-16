import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from detector import BBox, YoloSegment
import cv2
from pynput import keyboard


enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


if __name__ == '__main__':
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    detector = YoloSegment("yolov8n-seg.pt")
    #create video stream
    print('Starting Camera stream:......')
    cap = cv2.VideoCapture(0)
    while enable:
        ret, frame = cap.read()
        if ret:
            detector.eval(frame)
            
            cv2.imshow('frame',frame)
            cv2.waitKey(1)

    print('Closing Camera stream:')
    cap.release()
    cv2.destroyAllWindows()
    listener.join()