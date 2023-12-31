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
    pos[2] *= -1
    rot_mat = pose.rot_mat.copy()
    rot_mat[0,:3] *= -1
    rot_mat[1,:3] *= -1

    rot = Rotation.from_matrix(rot_mat).as_quat()
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

class ClassDisplayWindow:
    def __init__(self, bbox3d, text="Default"):
        x0,y0,z0 = bbox3d.getTL()
        x1,y1,z1 = bbox3d.getBR()

        cx = (x0+x1)/2
        cy = (y0+y1)/2
        cz = (z0+z1)/2
        center = np.array([[cx,cy,cz]]).T

        w = (x1-x0)
        l = (y1-y0)
        h = (z1-z0)

        #WINDOW
        rotz = Rotation.from_rotvec(np.array([0,0,np.pi/2])).as_matrix()
        rot = Rotation.from_rotvec(np.array([np.pi,0,0])).as_matrix()

        rotz = Rotation.from_rotvec(np.array([0,np.pi/2,0])).as_matrix() @ rotz
        rot = Rotation.from_rotvec(np.array([0,np.pi/2,0])).as_matrix() @ rot
        self.window = Pose(rotz, center + np.array([[0,l,0]]).T)

        
        self.text   = Pose(rot, center + np.array([[-3e-2,l,0]]).T)
        self.str = text

        self.objs_pose = [self.window, self.text]

    def create_render(self):
        self.objs = []
        scale = [0.1,0.12,1e-3]
        rgba = [80/255,80/255,140/255,1]
        self.objs.append(pose2render(self.window, scale, rgba, 'capsule'))

        scale = [1,1,1]
        rgba = [1,1,1,1]
        obj = pose2render(self.text, scale, rgba, 'text')
        obj.text = self.str
        obj.font_size = 0.4
        self.objs.append(obj)

        return self.objs


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

class DetBox:
    def __init__(self,bbox3d,thickness=1e-1):
        self.display_window = ClassDisplayWindow(bbox3d,bbox3d.name)
        self.box = RenderBBox(bbox3d,thickness)

        self.objs_pose = self.display_window.objs_pose + self.box.objs_pose
        self.objs = []

    def create_render(self):
        objs1 = self.display_window.create_render()
        objs2 = self.box.create_render()

        self.objs = objs1 + objs2
        return self.objs

    def setWindowPose(self, pose):
        self.display_window.objs = setMultiObjectPose(self.display_window.objs, self.display_window.objs_pose, pose)

        self.objs = self.box.objs + self.display_window.objs
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
    

class HandRenderer:
    def __init__(self,renderer):
        self.renderer = renderer

        self.lefthand_ids = []
        self.righthand_ids = []

        self.lefthand_objs = []
        self.righthand_objs = []
    def visualize_lefthand(self, left_pos):
        N = left_pos.shape[0]
        if len(self.lefthand_ids) == 0:
            for i in range(N):
                pos = left_pos[i,:].tolist()
                # print(pos)
                rotation = [0,0,0,1]
                scale = [0.01,0.01,0.01]
                rgba = [1,1,1,1]
                pos[2] *= -1

                self.lefthand_objs.append(RenderObject('sphere',pos,rotation,scale,rgba))
            self.lefthand_ids = self.renderer.addPrimObjects(self.lefthand_objs)
        else:
            for i in range(N):
                pos = left_pos[i,:].tolist()
                pos[2] *= -1
                self.lefthand_objs[i].pos = pos
            self.renderer.transformObjs(self.lefthand_ids,self.lefthand_objs)
    def visualize_righthand(self, right_pos):
        N = right_pos.shape[0]
        if len(self.righthand_ids) == 0:
            for i in range(N):
                pos = right_pos[i,:].tolist()
                # print(pos)
                rotation = [0,0,0,1]
                scale = [0.01,0.01,0.01]
                rgba = [1,1,1,1]
                pos[2] *= -1

                self.righthand_objs.append(RenderObject('sphere',pos,rotation,scale,rgba))
            self.righthand_ids = self.renderer.addPrimObjects(self.righthand_objs)
        else:
            for i in range(N):
                pos = right_pos[i,:].tolist()
                pos[2] *= -1
                self.righthand_objs[i].pos = pos
            self.renderer.transformObjs(self.righthand_ids,self.righthand_objs)