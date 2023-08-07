import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import BBox


class KalmanFilter:
    '''
    x_k+1 = A*x_k + N(0,Q)
    y_k   = C*x_k + N(0,R)
    '''
    def __init__(self, A, C, Q, R, obj_id):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.id = obj_id
        
        self.state = None
        self.cov   = None
    
    def initialize(self, state0, cov0):
        self.state = state0
        self.cov   = cov0

    def predict(self):
        self.state  = self.A @ self.state
        self.cov = self.A @ self.cov @ self.A.T + self.Q

    def update(self, meas):
        K = self.cov @ self.C.T @ np.linalg.inv(self.C @ self.cov @ self.C.T + self.R)
        self.state  = self.state + K @ (meas - self.C @ self.state)
        self.cov = (np.eye(self.cov.shape[0]) - K @ self.C) @ self.cov
        
    def drawState(self, image):
        bbox = self.state_to_bbox(self.state)
        bbox.name = f'ID:{self.id}'
        
        height,width,_ = image.shape
        thick = int((height+width)//900)
        cv2.rectangle(image, (int(bbox.x0),int(bbox.y0)), (int(bbox.x1),int(bbox.y1)), (255,0,0))
        cv2.putText(image, bbox.name, (int(bbox.x0),int(bbox.y0)-12), 0, 1e-3*height, (255,0,0), thick//3)
        return image

    def getAllCorners(self):
        bbox = self.state_to_bbox(self.state)
        return bbox.getAllCorners()

    def state_to_bbox(self, state):
        cx = state[0,0]
        cy = state[1,0]
        s  = state[2,0]
        r  = state[3,0]
        
        w  = r*s
        h  = w/r
        
        xTL = cx - w/2
        xBR = cx + w/2

        yTL = cy - h/2
        yBR = cy + h/2

        return BBox(xTL,yTL,xBR,yBR,'unknown')

class TrackerObj(KalmanFilter):
    def __init__(self, A, C, Q, R, obj_id=0, age=0, is_matched_before=False):
        super().__init__(A, C, Q, R, obj_id) 
        self.is_matched_before  = is_matched_before 
        self.age = age 

class MultiObjectTracker:
    def __init__(self) -> None:
        self.id_ctr = 0
        self.objs = []
        self.conf = 0.95 

        self.gate_matrix_thresh1 = 500
        self.gate_matrix_thresh2 = 0.85
        self.MAX_AGE = 30

    def initialize_object(self, bbox):
        '''
        Description:
        ------------
            Convert a BBox into a new TrackObj, which represents a new track
            and appends it to the current list of tracked objects.

        Parameters:
        -----------
            bbox: BBox that is detected by an object detector. 

        Returns:
        --------
            None 
        '''
        track_vector = self.bbox_to_state(bbox)
        track_cov    = np.eye(8)
        
        #assign id
        id_ = self.id_ctr
        self.id_ctr += 1
        
        A = np.eye(8)
        #first 3 rows, last 3 column
        A[:4,4:] = np.eye(4)
        
        obj = TrackerObj(A=A, C=np.eye(8), Q=np.eye(8), R=1e-3*np.eye(8),obj_id=id_)
        obj.initialize(track_vector, track_cov)
        self.objs.append(obj)

    def track_boxes(self, bboxes, mode="2d"):
        '''
        Description:
        ------------
        Matches new detections to current tracks. Removes
        or updates the tracks as needed.

        Parameters:
        -----------
            bboxes: list[BBox] - A list of bounding boxes detected
            mode: str - Determines whether the bounding boxes are 2D or 3d
            and whether to perform 2D or 3D object tracking

        Returns:
        --------
            None
        '''
        #predict obj phase
        for obj in self.objs:
            obj.predict()        
        
        #associate
        obj_matched, det_matched = self.match_track_and_detections(
                bboxes, self._mahalanobis_dist_2d if mode == "2d" else self._mahalanobis_dist_3d)

        #any unmatched, just leave out....
        #perform update step 
        objs_new = []
        for i in range(len(obj_matched)):
            obj_idx = obj_matched[i]
            det_idx = det_matched[i]
            
            det = self.bbox_to_state(bboxes[det_idx])
            det[4:] = self.calc_velocity(det, self.objs[obj_idx].state)
            
            self.objs[obj_idx].update(det)
            objs_new.append(self.objs[i])
        
        #filter only objects that got tracked
        self.objs = objs_new
        
        #initialize any unmatched detections as a new track
        for i in range(len(bboxes)):
            if i not in det_matched:
                self.initialize_object(bboxes[i])

    def match_track_and_detections(self, bboxes, mh_dist):
        predicted_boxes = []
        for obj in self.objs:
            bbox = obj.state_to_bbox(obj.state)
            predicted_boxes.append(bbox)

        m = len(predicted_boxes)
        n = len(bboxes)
        cost_matrix = np.zeros((m,n))
        for pred_idx in range(m):
            for meas_idx in range(n):
                cost_matrix[pred_idx,meas_idx] = mh_dist(
                        self.bbox_to_state(predicted_boxes[pred_idx]), 
                        self.bbox_to_state(bboxes[meas_idx]),
                        self.objs[pred_idx])
        
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        obj_matched = []
        det_matched = []
        for i in range(len(row_ind)):
            obj_idx = row_ind[i]
            det_idx = col_ind[i]
            if cost_matrix[obj_idx,det_idx] >= self.gate_matrix_thresh1:
                obj_matched.append(obj_idx)
                det_matched.append(det_idx)

        return obj_matched, det_matched

    def _mahalanobis_dist_2d(self, tracks_state, detected_state, obj):
        S = (obj.C.T @ obj.cov @ obj.C + obj.R)[0:4,0:4]
        mh = ((detected_state - tracks_state).T @ np.linalg.inv(S) @ (detected_state - tracks_state))[0,0]
        return mh

    def _mahalanobis_dist_3d(self, tracks_state, detected_state, obj):
        S = (obj.C.T @ obj.cov @ obj.C + obj.R)#[0:4,0:4]
        mh = ((detected_state - tracks_state).T @ np.linalg.inv(S) @ (detected_state - tracks_state))[0,0]
        return mh

    def bbox_to_state(self, bbox):
        xTL, yTL = bbox.getTL()
        xBR, yBR = bbox.getBR()
        
        cx = (xTL + xBR) / 2
        cy = (yTL + yBR) / 2
        
        r = (xBR - xTL) / (yBR - yTL)
        s = (yBR - yTL)
        
        return np.array([[cx,cy,s,r,0,0,0,0]]).T

    def calc_velocity(self, curr_state, prev_state):
        vel = curr_state[:4] - prev_state[:4]
        return vel