#------------------------------------------------------------------------------
# async_processor.py
# 
# Contains functions for creating a distributed framework to handle data 
# streaming and image processing without losing any streaming data.
# The key bottleneck is that we can potentially miss frames doing a simple
# read data and process in a while loop.
# Additionally, we want to allow for multiple instances of object detectors to
# speed up processing of data.
#------------------------------------------------------------------------------
from detector import YoloDetector
import multiprocessing as mp
import time
import cv2
# from multiprocessing import Process, Queue

'''
Current Idea:
--------------
    one main process will call AsyncManager

    man = AsyncManager()

    def foo():
        .... run YOLO

    #adding processes
    #adding two YOLO
    man.addProcess(foo)
    man.addProcess(foo)

    #can no longer do .addProcess()
    #everything starts now
    man.start()
'''

class AsyncManager:
    def __init__(self):
        self.processes = []

        self.inDetectorQueue = mp.Queue()
        self.outDetectorQueue = mp.Queue()

    def addProcess(self,fn, *arg):
        proc = mp.Process(target=fn,args=arg)
        self.processes.append(proc)
    def start(self):
        for process in self.processes:
            process.start()
    def stop(self):
        print('Killing Processes')
        for process in self.processes:
            process.kill()
        time.sleep(1) # wait a bit before reaping reosurces from each process
        print('Closing Processes')
        for process in self.processes:
            process.close()

    def getInDetectorQueue(self):
        return self.inDetectorQueue
    def getOutDetectorQueue(self):
        return self.outDetectorQueue
    
def YoloProcess(model_file,in_queue, out_queue):
    detector = YoloDetector(model_file)
    while 1:
        if not in_queue.empty():
            image = in_queue.get()
            bboxes = detector.eval(image)
            if len(bboxes) > 0:
                out_queue.put((image,bboxes))
    #run until it gets killed

def DisplayProcess(out_detector_queue):
    while 1:
        if not out_detector_queue.empty():
            image,bboxes = out_detector_queue.get()
            for bbox in bboxes:
                image = bbox.drawBox(image)
            cv2.imshow('frame',image)
            cv2.waitKey(1)