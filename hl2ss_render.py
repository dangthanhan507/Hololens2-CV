#------------------------------------------------------------------------------
# hl2ss_render.py
# 
# This is hl2ss wrapper for rendering on the hololens visor. This will allow
# us to start/stop rendering software for the visor and also render any 
# primitive we want on the visor with access to color, primitives, and 
# transform. 
#------------------------------------------------------------------------------

from hl2ss_stream import HOST_IP, ROOT_PATH
import os
import sys
sys.path.append(os.path.join(ROOT_PATH,'hl2ss','viewer'))

import hl2ss
import hl2ss_rus
import numpy as np

def getObjectType(object):
    if object == "cube":
        return hl2ss_rus.PrimitiveType.Cube
    elif object == "capsule":
        return hl2ss_rus.PrimitiveType.Capsule
    elif object == "cylinder":
        return hl2ss_rus.PrimitiveType.Cylinder
    elif object == "sphere":
        return hl2ss_rus.PrimitiveType.Sphere
    elif object == "plane":
        return hl2ss_rus.PrimitiveType.Plane
    elif object == "quad":
        return hl2ss_rus.PrimitiveType.Quad

class RenderObject:
    def __init__(self, object, pos ,rot, scale, rgba):
        '''
            pos: list size 3 [x,y,z]
            rot: quaternion list size 4 [x,y,z,w]
            scale: list size 3 [sx,sy,sz]
            rgba: list size 4 [r,g,b,a]
        '''
        self.object = getObjectType(object)
        self.pos = pos
        self.rot = rot #quaternion
        self.scale = scale
        self.rgba = rgba

#NOTE: look at hl2ss_rus for primitives
#NOTE: plane primitive has issues
class Hl2ssRender:
    def __init__(self):
        self.objs = []
        self.ipc = hl2ss.ipc_umq(HOST_IP, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)

    def start(self):
        self.ipc.open()
    def stop(self):
        self.ipc.close()

    def addPrimObject(self, object, pos, rot, scale, rgba):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()

        #create obj
        display_list.create_primitive(getObjectType(object))
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
        display_list.set_world_transform(0, pos, rot, scale)
        display_list.set_color(0, rgba)
        display_list.set_active(0, hl2ss_rus.ActiveState.Active)
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
        display_list.end_display_list()
        self.ipc.push(display_list)
        results = self.ipc.pull(display_list)

        self.objs.append(results[1])
        return results[1]
    def addPrimObject(self, render_object):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()

        #create obj
        display_list.create_primitive(render_object.object)
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
        display_list.set_world_transform(0, render_object.pos, render_object.rot, render_object.scale)
        display_list.set_color(0, render_object.rgba)
        display_list.set_active(0, hl2ss_rus.ActiveState.Active)
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
        display_list.end_display_list()
        self.ipc.push(display_list)
        results = self.ipc.pull(display_list)

        self.objs.append(results[1])
        return results[1]
    
    def addPrimObjects(self, render_objects):
        '''
            render_objects: list of RenderObjects
        '''
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list() #cmd 0

        N = len(render_objects)
        for i in range(len(render_objects)):
            render_obj = render_objects[i]
            pos = render_obj.pos
            rot = render_obj.rot
            scale = render_obj.scale
            rgba = render_obj.rgba
            objType = render_obj.object

            display_list.create_primitive(objType) # cmd 1 + 5*i
            display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
            display_list.set_world_transform(0, pos, rot, scale)
            display_list.set_color(0, rgba)
            display_list.set_active(0, hl2ss_rus.ActiveState.Active)
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
        display_list.end_display_list()
        self.ipc.push(display_list)
        results = self.ipc.pull(display_list)
        # print(np.array(results).astype(np.float32).reshape((-1,1)))

        offset = 0
        while results[offset] == 1:
            offset += 1
            if offset >= len(results):
                offset = 0
                break
        cmd_idxs = list(range(offset,offset+5*N,5)) # 1, 6, 11,.... listing all idx of added obj
        

        object_ids = [results[idx] for idx in cmd_idxs]
        self.objs = self.objs + object_ids

        return object_ids
    def transformObj(self, object_id, render_obj):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        display_list.set_world_transform(object_id, render_obj.pos, render_obj.rot, render_obj.scale)
        display_list.end_display_list()
        self.ipc.push(display_list)
        results = self.ipc.pull(display_list)
        return results
    def transformObjs(self, object_ids, render_objs):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        for i in range(len(object_ids)):
            object_id = object_ids[i]
            render_obj = render_objs[i]
            display_list.set_world_transform(object_id, render_obj.pos, render_obj.rot, render_obj.scale)
        display_list.end_display_list()
        self.ipc.push(display_list)
        results = self.ipc.pull(display_list)
        return results
    
    def removePrimObject(self, object_id):
        if object_id in self.objs:
            self.objs.remove(object_id)

            display_list = hl2ss_rus.command_buffer()
            display_list.begin_display_list()
            display_list.remove(object_id)
            display_list.end_display_list()
            self.ipc.push(display_list)



    def clear(self):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        display_list.remove_all()
        display_list.end_display_list()
        self.ipc.push(display_list)
        self.ipc.pull(display_list)
        self.objs = []
