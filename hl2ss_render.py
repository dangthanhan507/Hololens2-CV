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

    def getObject(self, object):
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

    def addPrimObject(self, object, pos, rot, scale, rgba):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()

        #create obj
        display_list.create_primitive(self.getObject(object))
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
        display_list.set_world_transform(0, pos, rot, scale)
        display_list.set_color(0, rgba)
        display_list.set_active(0, hl2ss_rus.ActiveState.Active)
        display_list.end_display_list()

        self.ipc.push(display_list)
        results = self.ipc.pull(display_list)

        self.objs.append(results[1])
        return results[1]
    
    def removePrimObject(self, object_id, pos, rot, scale):
        if object_id in self.objs:
            self.objs.remove(object_id)

            display_list = hl2ss_rus.command_buffer()
            display_list.begin_display_list()
            display_list.remove(object_id)
            display_list.remove_display_list()
            self.ipc.push(display_list)



    def clear(self):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        display_list.remove_all()
        display_list.end_display_list()
        self.ipc.push(display_list)
