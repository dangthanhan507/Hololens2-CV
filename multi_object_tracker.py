import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import BBox, BBox3D


class KalmanFilter:
    '''
    x_k+1 = A*x_k + N(0,Q)
    y_k   = C*x_k + N(0,R)
    '''
    def __init__(self, A, C, Q, R):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        
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
        
class TrackerObj(KalmanFilter):
    def __init__(self, A, C, Q, R, obj_id=0, age=0, is_matched_before=False):
        super().__init__(A, C, Q, R) 
        self.is_matched_before  = is_matched_before 
        self.id = obj_id
        self.age = age
    
    def drawState(self, image):
        bbox = self.state_to_bbox()
        bbox.name = f'ID:{self.id}'
        
        height,width,_ = image.shape
        thick = int((height+width)//900)
        cv2.rectangle(image, (int(bbox.x0),int(bbox.y0)), (int(bbox.x1),int(bbox.y1)), (255,0,0))
        cv2.putText(image, bbox.name, (int(bbox.x0),int(bbox.y0)-12), 0, 1e-3*height, (255,0,0), thick//3)
        print(f"drawing state: {(int(bbox.x0),int(bbox.y0)), (int(bbox.x1),int(bbox.y1))}")
        return image

    def getAllCorners(self):
        bbox = self.state_to_bbox()
        return bbox.getAllCorners()

    def state_to_bbox(self):
        # TODO: remove reliance on state parameter, use self.state instead!!!
        cx = self.state[0,0]
        cy = self.state[1,0]
        cz = self.state[2,0]
        l = self.state[3,0]
        w = self.state[4,0]
        h = self.state[5,0]
        
        xTL = cx - w/2
        xBR = cx + w/2
        yTL = cy - l/2
        yBR = cy + l/2
        zTL = cz - h/2
        zBR = cz + h/2

        return BBox3D(xTL, yTL, zTL, xBR, yBR, zBR, name=f"ID:{self.id}")

    def increment_age(self):
        self.age += 1

class MultiObjectTracker:
    def __init__(self) -> None:
        self.id_ctr = 0
        self.objs = []
        self.conf = 0.95 

        self.gate_matrix_thresh1 = 10 

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
        track_cov    = np.eye(9)
        
        #assign id
        id_ = self.id_ctr
        self.id_ctr += 1
        
        #first 3 rows, last 3 column
        # A = np.eye(8)
        # A[:4,4:] = np.eye(4)
        A = np.eye(9)
        A[:3, -3:] = np.eye(3)
        
        obj = TrackerObj(A=A, C=np.eye(9), Q=np.eye(9), R=1e-3*np.eye(9),obj_id=id_)
        obj.initialize(track_vector, track_cov)
        self.objs.append(obj)

    def drawTracks(self, image):
        for obj in self.objs:
            image = obj.drawState(image)
        return image

    def track_boxes(self, bboxes):
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
        print("------------------")
        print("Num of current objs:", len(self.objs))
        #predict obj phase
        for obj in self.objs:
            obj.predict()        
        
        #associate
        obj_matched, det_matched = self.match_tracks_and_detections(bboxes)
        print("Number of tracks matched:", len(obj_matched))

        #any unmatched, just leave out....
        #perform update step 
        objs_new = []
        for i in range(len(obj_matched)):
            obj_idx = obj_matched[i]
            det_idx = det_matched[i]
            
            det = self.bbox_to_state(bboxes[det_idx])
            det[-3:] = self.calc_velocity()
            
            self.objs[obj_idx].update(det)
            objs_new.append(self.objs[i])
        for i in range(len(self.objs)):
            if i not in obj_matched:
                objs_new.append(self.objs[i])
        
        #filter only objects that got tracked
        self.objs = objs_new
        
        #initialize any unmatched detections as a new track
        for i in range(len(bboxes)):
            if i not in det_matched:
                self.initialize_object(bboxes[i])


        # update objs
        print("| Num of objects after iteration:", len(self.objs))
        for obj in self.objs:
            obj.increment_age()
            print("| id:", obj.id)
            print("| corners:", obj.state_to_bbox().getAllCorners())


    def get_bbox_3d_pts(self):
        pts3d = np.zeros((3, len(self.objs) * 2))
        for i in range(len(self.objs)):
            pts3d[:, 2*i:2*i+2] = self.objs[i].state_to_bbox().getAllCorners().reshape((3,2), order='F')
        return pts3d

    def match_tracks_and_detections(self, bboxes):
        predicted_boxes = []
        for obj in self.objs:
            bbox = obj.state_to_bbox()
            predicted_boxes.append(bbox)

        m = len(predicted_boxes)
        n = len(bboxes)
        cost_matrix = np.zeros((m,n))
        for pred_idx in range(m):
            for meas_idx in range(n):
                cost_matrix[pred_idx,meas_idx] = self._mahalanobis_dist_3d(
                        self.bbox_to_state(predicted_boxes[pred_idx]), 
                        self.bbox_to_state(bboxes[meas_idx]),
                        self.objs[pred_idx])

                #SANITY CHECK TO MAKE SURE MAHALANOBIS DISTANCE WORKS AS INTENDED
                # e_dist = ((predicted_boxes[pred_idx].getAllCorners() - bboxes[meas_idx].getAllCorners()) ** 2).sum()
                # if e_dist < 50:
                #     print("| Could be plausible match, m dist. is", cost_matrix[pred_idx, meas_idx], "|")
                #     print(f"| Pred: {predicted_boxes[pred_idx].getAllCorners()}  |")
                #     print(f"| Detected: {bboxes[meas_idx].getAllCorners()}  |")

        
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        obj_matched = []
        det_matched = []
        for i in range(len(row_ind)):
            obj_idx = row_ind[i]
            det_idx = col_ind[i]
            if cost_matrix[obj_idx,det_idx] <= self.gate_matrix_thresh1:
                obj_matched.append(obj_idx)
                det_matched.append(det_idx)

        return obj_matched, det_matched

    def _mahalanobis_dist_3d(self, tracks_state, detected_state, obj):
        S = (obj.C.T @ obj.cov @ obj.C + obj.R)#[0:4,0:4]
        mh = ((detected_state - tracks_state).T @ np.linalg.inv(S) @ (detected_state - tracks_state))[0,0]
        return mh

    def bbox_to_state(self, bbox):
        xTL, yTL, zTL = bbox.getTL()
        xBR, yBR, zBR = bbox.getBR()
        
        cx = (xTL + xBR) / 2
        cy = (yTL + yBR) / 2
        cz = (zTL + zBR) / 2
        l = yBR - yTL
        w = xBR - xTL
        h = zBR - zTL
        
        return np.array([[cx,cy,cz,l,w,h,0,0,0]]).T

    def calc_velocity(self):
        return np.zeros((3,1))
    
    def getBBoxes(self):
        return [obj.state_to_bbox() for obj in self.objs]


class InteractableObject(TrackerObj):
    def __init__(self):
        pass

class InteractableMOT(MultiObjectTracker):
    '''
        Tracking Logic:
        ===============

        Hand-Tracking:
        ==============
            left hand: [0,.... maxId] 0 means holds nothing... anything else means it's holding that object
            right hand: [0,.... maxId]

    '''
    def __init__(self):
        #TODO: make some of these easily set by user
        super(MultiObjectTracker).__init__()

        self.left_pos = None
        self.left_holding = 0

        self.right_pos = None
        self.right_holding = 0


    def parse_boxes(self, bboxes3d):
        '''
            Ignore any bboxes close to hand when it is already grabbed (most likely it is the object grabbed).
        '''
        if self.left_holding != 0 or self.right_holding != 0:
            newBboxes3d = []
            for bbox3d in bboxes3d:
                if np.linalg.norm(self.left_pos - bbox3d.getCenter()) < 0.1:
                    continue
                if np.linalg.norm(self.right_pos - bbox3d.getCenter()) < 0.1:
                    continue

                newBboxes3d.append(bbox3d)
        else:
            newBboxes3d = bboxes3d
        return newBboxes3d
    
    def ungrab_cost(self, pts3d):
        '''
            solve optimization problem to get the normal vector corresponding to the plane
        '''
        mean_pts = pts3d.mean(axis=1)
        prel = pts3d - mean_pts.reshape((3,1))
        W = prel @ prel.T
        w, V = np.linalg.eigh(W)
        R = np.fliplr(V)
        normal_vec = R[:,2].reshape((3,1))

        #returns the sum of the dot products between normal vec and its pts (should be near zero)
        return np.abs(normal_vec.T @ prel).sum()
    
    def check_letgo(self, left_hand, right_hand):
        if self.left_holding != 0 and self.ungrab_cost(left_hand) < 0.1:
            #set object grab to nothing
            self.left_holding = 0

        if self.right_holding != 0 and self.ungrab_cost(right_hand) < 0.1:
            #set object grab to nothing
            self.right_holding = 0

    def track_hands(self, left_hand, right_hand):

        #update hand positions
        if left_hand is not None:
            self.left_pos = left_hand.T.mean(axis=1).reshape((3,1))
        if right_hand is not None:
            self.right_pos = right_hand.T.mean(axis=1).reshape((3,1))

        #update tracked detections
        if self.left_holding == 0 or self.right_holding == 0:
            for i in range(len(self.objs)):
                obj = self.objs[i]
                if self.left_holding == 0 and self.left_pos is not None and \
                    np.linalg.norm(self.left_pos - obj.state_to_bbox().getCenter()) < 0.1:
                    self.left_holding = obj.id
                    break
                elif self.right_holding == 0 and self.right_pos is not None and \
                    np.linalg.norm(self.right_pos - obj.state_to_bbox().getCenter()) < 0.1:
                    self.right_holding = obj.id
                    break
        
        if self.left_holding != 0 or self.right_holding != 0:
            pass


    def trackInteraction(self, left_hand, right_hand):
        '''
            Description:
            -----------
            in addition

            Parameters:
            -----------
            @params left_hand: (3xN) set of points representing left hand detected or it can be a None value
            @params right_hand: (3xN) set of points representing right hand detected or it can be a None value
        '''
        if left_hand is not None:
            pass
        if right_hand is not None:
            pass