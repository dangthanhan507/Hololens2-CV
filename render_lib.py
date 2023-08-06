#------------------------------------------------------------------------------
# render_lib.py
# 
# This is a file containing all of the structures for rendering using 
# hl2ss_render. This should encapsulate all of the code we envision using
# for rendering objects. 
#------------------------------------------------------------------------------
from hl2ss_render import RenderObject
from detector import BBox3D
from scipy.spatial.transform import Rotation
import numpy as np

class Pose:
    def __init__(self, rotm, tvec):
        self.rot_mat = rotm
        self.t_vec = tvec

def pose2render(pose, scale, rgba, type):
    
    pos = pose.t_vec.flatten().tolist()
    rot = Rotation.from_matrix(pose.rot_mat).as_quat()
    return RenderObject(type, pos, rot, scale, rgba)

def setMultiObjectPose(objs,objs_pose, world_pose):
    '''
    '''
    world_rot = world_pose[:3,:3]
    world_tvec=  world_pose[:3,-1].reshape((3,1))

    for i in range(len(objs_pose)):
        rot_obj = objs_pose[i].rot_mat
        tvec_obj = objs_pose[i].t_vec

        pose = Pose(world_rot @ rot_obj, (world_rot @ tvec_obj) + world_tvec)
        objs[i] = pose2render(pose,objs[i].scale, objs[i].rgba, objs[i].object)
    
    return objs

class RenderBBox:
    def __init__(self, bbox3d, thickness=1e-1, rgba = [1,1,1,1]):
        '''
        bbox3d: type BBox3D

        create an axis-aligned bounding box with its object frame in the center
        '''
        x0,y0,z0 = bbox3d.getTL()
        x1,y1,z1 = bbox3d.getBR()

        cx = (x0+x1)/2
        cy = (y0+y1)/2
        cz = (z0+z1)/2
        center = np.array([[cx,cy,cz]]).T

        w = (x1-x0)
        l = (y1-y0)
        h = (z1-z0)

        self.w = w
        self.l = l
        self.h = h


        self.rgba = rgba
        self.thickness = thickness
        #12 primitives associated with 1 bbox
        # edges of the bottom face (4), edges of top face (4), side pillars (4)


        #BOTTOM FACE
        self.bot0 = Pose(np.eye(3), center + np.array([[w/2,-l/2,0]]).T)
        self.bot1 = Pose(np.eye(3), center + np.array([[-w/2,-l/2,0]]).T)
        self.bot2 = Pose(np.eye(3), center + np.array([[0,-l/2,h/2]]).T)
        self.bot3 = Pose(np.eye(3), center + np.array([[0,-l/2,-h/2]]).T)

        #TOP FACE
        self.top0 = Pose(np.eye(3), center + np.array([[w/2,l/2,0]]).T)
        self.top1 = Pose(np.eye(3), center + np.array([[-w/2,l/2,0]]).T)
        self.top2 = Pose(np.eye(3), center + np.array([[0,l/2,h/2]]).T)
        self.top3 = Pose(np.eye(3), center + np.array([[0,l/2,-h/2]]).T)

        #SIDE PILLARS
        self.side0 = Pose(np.eye(3), center + np.array([[w/2,0,h/2]]).T)
        self.side1 = Pose(np.eye(3), center + np.array([[w/2,0,-h/2]]).T)
        self.side2 = Pose(np.eye(3), center + np.array([[-w/2,0,-h/2]]).T)
        self.side3 = Pose(np.eye(3), center + np.array([[-w/2,0,h/2]]).T)

        self.objs_pose = [self.bot0, self.bot1, self.bot2, self.bot3,
                          self.top0, self.top1, self.top2, self.top3,
                          self.side0, self.side1, self.side2, self.side3]
        
        self.objs = []
        self.obj_ids = []

    def create_render(self):
        self.objs = []

        scale = [self.thickness,self.thickness,self.h]
        self.objs.append( pose2render(self.bot0, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.bot1, scale, self.rgba, 'cube') )
        scale = [self.w,self.thickness,self.thickness]
        self.objs.append( pose2render(self.bot2, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.bot3, scale, self.rgba, 'cube') )

        scale = [self.thickness,self.thickness,self.h]
        self.objs.append( pose2render(self.top0, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.top1, scale, self.rgba, 'cube') )
        scale = [self.w,self.thickness,self.thickness]
        self.objs.append( pose2render(self.top2, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.top3, scale, self.rgba, 'cube') )

        scale = [self.thickness,self.l,self.thickness]
        self.objs.append( pose2render(self.side0, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.side1, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.side2, scale, self.rgba, 'cube') )
        self.objs.append( pose2render(self.side3, scale, self.rgba, 'cube') )

        return self.objs
        

class CoordinateFrame:
    def __init__(self, offset, size):
        self.offset = offset
        self.size = size

    def create_render(self):
        #draw coordinate frame
        size = self.size

        pos = self.offset.copy()
        pos[0] += size/2
        rotation = [0,0,0,1]
        scale = [size,1e-2,1e-2]
        rgba = [1,0,0,1]

        xaxis = RenderObject('cube', pos, rotation, scale, rgba)

        pos = self.offset.copy()
        pos[1] += size/2
        scale = [1e-2,size,1e-2]
        rgba = [0,1,0,1]
        yaxis = RenderObject('cube', pos, rotation, scale, rgba)

        pos = self.offset.copy()
        pos[2] += size/2
        scale = [1e-2,1e-2,size]
        rgba = [0,0,1,1]
        zaxis = RenderObject('cube', pos, rotation, scale, rgba)

        return [xaxis,yaxis,zaxis]