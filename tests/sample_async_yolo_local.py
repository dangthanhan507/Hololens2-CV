#------------------------------------------------------------------------------
# sample_async_yolo_local.py
# 
# EXPERIMENTAL code attempting to separate YOLO from stream capture.
# this is done on local but should be transferred to HL2ss Stream 
#
#
# NOTE: we want the parent process (the main of this script) to run the streaming
# This is important because we want one centralized process/thread to download
# and upload data to the Hololens. We want to control and keep this thread 
# for as long as possible. Everything else is just a child of the main streaming
# process the server provides. In other words, this script should attempt to have
# the parent process by the webcam streamer and the yolo be the child process
# with multiprocessing Manager working with the shared memory between them.
# Another benefit to this is startin up multiple YOLO Queue's to speeed up
# rate at which images are processed.
#------------------------------------------------------------------------------

import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from async_processor import AsyncManager, YoloProcess, DisplayProcess
import cv2
from pynput import keyboard

#the magic library
import multiprocessing as mp





# KEYBOARD LISTENER SETUP
#-------------------------
enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable
#-------------------------

if __name__ == '__main__':
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    manager = AsyncManager()

    #add processes
    manager.addProcess(YoloProcess, "yolov8n.pt", manager.inDetectorQueue, manager.outDetectorQueue)
    manager.addProcess(YoloProcess, "yolov8n.pt", manager.inDetectorQueue, manager.outDetectorQueue)
    manager.addProcess(DisplayProcess,manager.outDetectorQueue)
    manager.start()
    print('Starting Camera stream:......')
    cap = cv2.VideoCapture(0)
    while enable:
        ret, frame = cap.read()
        if ret:
            manager.inDetectorQueue.put(frame)
    
    print('Closing Camera stream:')
    cap.release()
    manager.stop()

    listener.join()