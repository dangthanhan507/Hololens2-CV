#------------------------------------------------------------------------------
# detector.py
# 
# Adding a detector class to contain code wrapping object detector/segment under a 
# unified framework. 
#------------------------------------------------------------------------------

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class BBox3D:
    def __init__(self, x0, y0, z0, x1, y1, z1, name):
        '''
            extension of 2d boudning box into a 3d origin axis aligned bounding box.
            only requires two 3d points to represent it.

            in 2d case:
            -----------
            x0,y0 = getTL()
            x1,y1 = getBR()

            x0 < x1
            y0 < y1

            in 3d case we wish to have the same inequalities:
            =================================================
            x0,y0,z0 = getTL()
            x1,y1,z1 = getBR()
        '''
        self.name = name
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

    def getTL(self):
        return (self.x0, self.y0, self.z0)
    def getBR(self):
        return (self.x1, self.y1, self.z1)
    def getAllCorners(self):
        return np.array([self.x0, self.y0, self.z0, self.x1, self.y1, self.z1])

'''
Custom Bounding Box Class:
--------------------------
we want to abstract away the usage of yolo or another other models (mmdetection)
in the near future, so it is good to have a centralized way of representing the data structure
of the inference output.

this makes it easy to work with object tracking and any future extension.
'''
class BBox:
    def __init__(self, x0, y0, x1, y1, name):
        '''
        FORMAT:
        -------
            -> we follow top left bottom right format. (xyxy)
            -> (x0,y0) reprsents top left point of box
            -> (x1,y1) represents bottom right point of box
            -> with this in mind in the image context, we know that x0 < x1 and y0 < y1.
            -> we also have a name attached to distinguish between bounding boxes.

        Parameters:
        -----------
            x0: double
            x1: double
            y0: double
            y1: double
            name: string representing class name for model
        '''
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.name = name
    def getTL(self):
        # get topleft xy point of bounding box
        return (self.x0,self.y0)
    def getBR(self):
        # get bottom right xy point of bounding box
        return (self.x1,self.y1)
    
    def drawBox(self, image):
        '''
        Description:
        -------------
        visualizes a bounding box by drawing it on an image.
        this happens in place as to not use up more memory.

        Parameters:
        ------------
            image: (M,N,3) np.uint8 array representing RGB image.

        Returns:
        --------
            drawnImage: (M,N,3) np.uint8 array representing RGB image with a bounding box drawn on it.
        '''
        height,width,_ = image.shape
        thick = int((height+width)//900)
        cv2.rectangle(image, (int(self.x0),int(self.y0)), (int(self.x1),int(self.y1)), (255,0,0))
        cv2.putText(image, self.name, (int(self.x0),int(self.y0)-12), 0, 1e-3*height, (255,0,0), thick//3)
        return image

    def getAllCorners(self):
        '''
        Description:
        -------------
        Returns all coordinate values of the BBox's 4 corners

        Returns:
        --------
            tuple of len. 4, each is np.float64
        '''
        return self.x0, self.x1, self.y0, self.y1

'''
YoloDetector:
-------------
Quick class wrapper around getting the results of yolo model and outputting it in the desirable BBox format.
Unifies the usage of yolo under a class so we can do more with yolo. Also serves as an example for future usage
with other models in case this gets extended.

'''
class YoloDetector:
    def __init__(self, model_file):
        if not model_file.endswith('.pt'):
            raise ValueError("File string should end with .pt")
        
        self.model = YOLO(model_file)
    def eval(self, image, filter_cls = []):
        '''
        Description:
        -------------
        Evaluates a colored image using yolo and takes out detection.

        Parameters:
        ------------
            image: (M,N,3) np.uint8 array representing RGB image.
            filter_cls: [String] representing list of strings that we would filter our results by 
                        to only focus on specific classes on a model.
        Returns:
        --------

            box_ret: [BBox] returns list of BBox containing results
        '''
        results = self.model.predict(image,conf=0.5)
        box_ret = []
        '''
        Yolov8 NOTE:
        -------------
        Model inference on yolov8 returns a results.

        Results contains boxes which is what we need.
        Boxes contains xyxy (top left to bottom right)

        '''
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0] # top left, bot right
                c = box.cls #class index

                #use class index to access name in models
                name = self.model.names[int(c)]

                #if we decide to filter boxes for a specific kind of class
                if len(filter_cls) > 0:
                    if name in filter_cls:
                        box_ret.append(BBox(xyxy[0],xyxy[1],xyxy[2],xyxy[3],name))
                else:
                    box_ret.append(BBox(xyxy[0],xyxy[1],xyxy[2],xyxy[3],name))
        return box_ret
    
class YoloSegment:
    def __init__(self, model_file):
        if not model_file.endswith('.pt'):
            raise ValueError("File string should end with .pt")
        
        self.model = YOLO(model_file)
    def eval(self, image, filter_cls=[]):
        '''
        Description:
        -------------
        Evaluates a colored image using yolo and extract segmentation and detection.

        Parameters:
        ------------
            image: (M,N,3) np.uint8 array representing RGB image.
            filter_cls: [String] representing list of strings that we would filter our results by 
                        to only focus on specific classes on a model.
        Returns:
        --------
            seg: (M,N) image of segmentations
            box_ret: [BBox] returns list of BBox containing results
        '''

        height, width, _ = image.shape
        results = self.model.predict(image,boxes=True,conf=0.25)
        # results = self.model.predict(image,boxes=True,conf=0.25,retina_masks=True)
        box_ret = []
        mask_ret = []
        for result in results:
            if result.masks is not None:
                (N,H,W) = result.masks.data.shape
                for n in range(N):
                    mask = result.masks.data[n,:,:]

                    box = result.boxes[n]
                    xyxy = box.xyxy[0] # top left, bot right
                    c = box.cls #class index

                    #use class index to access name in models
                    name = self.model.names[int(c)]

                    #if we decide to filter boxes for a specific kind of class
                    if len(filter_cls) > 0:
                        if name in filter_cls:
                            box_ret.append(BBox(xyxy[0],xyxy[1],xyxy[2],xyxy[3],name))
                            mask_ret.append(mask.cpu().numpy())
                    else:
                        box_ret.append(BBox(xyxy[0],xyxy[1],xyxy[2],xyxy[3],name))
                        mask_ret.append(mask.cpu().numpy())
        return mask_ret, box_ret
