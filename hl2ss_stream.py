#------------------------------------------------------------------------------
# hl2ss_stream.py
# 
# This is a stream wrapper for the hl2ss code. Abstracts away all of the 
# complexity involved with using the hl2ss code. This will be in the form of a 
# class where method calls will give us what we need for the sensor data  
# extracted from hl2ss. The details of the setup streaming code should not be 
# too relevant in this repo, so we can hardcode them in this file.
#------------------------------------------------------------------------------


ROOT_PATH = "../Hololens2-CV-Server/"

#library imports
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(ROOT_PATH,'hl2ss','viewer'))
import hl2ss_3dcv
import hl2ss_mp
import hl2ss_utilities

#hl2ss specific imports
import hl2ss

# hl2ss Settings --------------------------------------------------------------
HOST_IP = '192.168.43.19' #ipv4 of hololens

#path to hololens calibration data (extrinsics/intrinsics)
CALIB_PATH = os.path.join(ROOT_PATH,'hl2ss','calibration')

# PV camera params
PV_WIDTH = 640
PV_HEIGHT = 360
PV_FPS = 30
PV_PROFILE = hl2ss.VideoProfile.H265_MAIN
PV_BITRATE = hl2ss.get_video_codec_bitrate(PV_WIDTH, PV_HEIGHT, PV_FPS, hl2ss.get_video_codec_default_factor(PV_PROFILE))

#VLC params
VLC_PROFILE = hl2ss.VideoProfile.H265_MAIN
VLC_BITRATE = 1*1024*1024
# General Params

#### in seconds
BUFFER_LENGTH = 10

# Longthrow Constants
MAX_DEPTH = 3.0

#------------------------------------------------------------------------------
class Hl2ssData:
    def __init__(self, data_lt,data_pv, pv_intrinsics, pv_extrinsics, data_si=None, data_lf=None, data_rf=None):
        self.data_lt = data_lt
        self.data_pv = data_pv

        self.color_intrinsics = pv_intrinsics
        self.color_extrinsics = pv_extrinsics

        #optional adds

        ### spatial input
        self.data_si = SpatialInputData(data_si)
        ### vlc front cameras
        self.data_lf = data_lf
        self.data_rf = data_rf

class SpatialInputData:
    def __init__(self, data_si):
        
        self.si = None
        self.head_pose  = None
        self.hand_left  = None
        self.hand_right = None
        self.eye_ray    = None

        if not (data_si is None):
            print('Unpacking')
            self.si = hl2ss.unpack_si(data_si.payload)

            if self.si.is_valid_head_pose():
                self.head_pose = self.si.get_head_pose()
            if self.si.is_valid_hand_left():
                self.hand_left = hl2ss_utilities.si_unpack_hand(self.si.get_hand_left())
            if self.si.is_valid_hand_right():
                self.hand_right = hl2ss_utilities.si_unpack_hand(self.si.get_hand_right())
            if self.si.is_valid_eye_ray():
                self.eye_ray = self.si.get_eye_ray()
        else:
            self.si = None

'''
Hl2ssStreamWrapper Features:
    -> RGB stream
    -> Depth stream
    -> RGBD fusion

    -> Obtain associated pose with data
'''
class Hl2ssStreamWrapper:
    def __init__(self, opts = {}):
        '''
        Parameters:
        -----------

        opts: {"feature": bool} dictionary of feature strings to see what feature to enable.
                by default, we just enable RGB and Depth. However, other features can be used.


        Options: (list of options from opts you can use)
        ------------------------------------------------
            -> "spatial input"
        '''

        #storing optional feature booleans
        self.spatial_input_enable = False
        self.vlc_front_enable = False

        self.checkOptional(opts)

        #setup and store Longthrow sensor calibrations
        self.calib_lt = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, os.path.join(CALIB_PATH,"rm_depth_longthrow"))
        uv2xy = hl2ss_3dcv.compute_uv2xy(self.calib_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        self.xy1, self.lt_scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calib_lt.scale)

        #setup pv calibrations (placeholder)
        self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

        # stream setup producer
        self.producer = hl2ss_mp.producer()
        self.consumer = hl2ss_mp.consumer()
        self.manager = mp.Manager()

        # sink placeholder variables
        self.sink_pv = None
        self.sink_depth = None

        #optional sinks
        self.sink_si = None
        self.sink_vlclf = None
        self.sink_vlcrf = None

        self.running = False
    
    def checkOptional(self, opts):
        if "spatial_input" in opts:
            self.spatial_input_enable = opts["spatial_input"]
        if "vlc_front" in opts:
            self.vlc_front_enable = opts['vlc_front']
        
    def start(self):
        '''
            Start up the Hololens Streaming. Sets up all necessary producer/consumer
            and sinks to achieve streaming capabilities
        '''
        hl2ss.start_subsystem_pv(HOST_IP, hl2ss.StreamPort.PERSONAL_VIDEO)

        #configure profiles
        self.producer.configure_pv(True, HOST_IP, hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.ChunkSize.PERSONAL_VIDEO, hl2ss.StreamMode.MODE_1, PV_WIDTH, PV_HEIGHT, PV_FPS, PV_PROFILE, PV_BITRATE, 'rgb24')
        self.producer.configure_rm_depth_longthrow(True, HOST_IP, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.ChunkSize.RM_DEPTH_LONGTHROW, hl2ss.StreamMode.MODE_1, hl2ss.PngFilterMode.Paeth)
        if self.spatial_input_enable:
            self.producer.configure_si(HOST_IP, hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.ChunkSize.SPATIAL_INPUT)
        if self.vlc_front_enable:
            self.producer.configure_rm_vlc(True, HOST_IP, hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss.ChunkSize.RM_VLC, hl2ss.StreamMode.MODE_1, VLC_PROFILE, VLC_BITRATE)
            self.producer.configure_rm_vlc(True, HOST_IP, hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss.ChunkSize.RM_VLC, hl2ss.StreamMode.MODE_1, VLC_PROFILE, VLC_BITRATE)
        

        #initialize producer
        self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, PV_FPS * BUFFER_LENGTH)
        self.producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * BUFFER_LENGTH)
        if self.spatial_input_enable:
            self.producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.Parameters_SI.SAMPLE_RATE * BUFFER_LENGTH)
        if self.vlc_front_enable:
            self.producer.initialize(hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss.Parameters_RM_VLC.FPS * BUFFER_LENGTH)
            self.producer.initialize(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss.Parameters_RM_VLC.FPS * BUFFER_LENGTH)

        #start producer
        self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
        if self.spatial_input_enable:
            self.producer.start(hl2ss.StreamPort.SPATIAL_INPUT)
        if self.vlc_front_enable:
            self.producer.start(hl2ss.StreamPort.RM_VLC_LEFTFRONT)
            self.producer.start(hl2ss.StreamPort.RM_VLC_RIGHTFRONT)

        #setup consumer
        self.sink_pv = self.consumer.create_sink(self.producer, hl2ss.StreamPort.PERSONAL_VIDEO, self.manager, None)
        self.sink_depth = self.consumer.create_sink(self.producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, self.manager, ...)
        if self.spatial_input_enable:
            self.sink_si = self.consumer.create_sink(self.producer, hl2ss.StreamPort.SPATIAL_INPUT, self.manager, None)
        if self.vlc_front_enable:
            self.sink_vlclf = self.consumer.create_sink(self.producer, hl2ss.StreamPort.RM_VLC_LEFTFRONT, self.manager, None)
            self.sink_vlcrf = self.consumer.create_sink(self.producer, hl2ss.StreamPort.RM_VLC_RIGHTFRONT, self.manager, None)

        self.sink_pv.get_attach_response()
        self.sink_depth.get_attach_response()
        if self.spatial_input_enable:
            self.sink_si.get_attach_response()
        if self.vlc_front_enable:
            self.sink_vlclf.get_attach_response()
            self.sink_vlcrf.get_attach_response()

        self.running = True

    def stop(self):
        '''
            Stops the Hololens Streaming. Must call this to ensure we can re-run program.
            If not, resources will still be allocated on program exit on the Hololens side.
        '''
        #stop network streams
        self.sink_pv.detach()
        self.sink_depth.detach()
        if self.spatial_input_enable:
            self.sink_si.detach()
        if self.vlc_front_enable:
            self.sink_vlclf.detach()
            self.sink_vlcrf.detach()

        self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
        if self.spatial_input_enable:
            self.producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
        if self.vlc_front_enable:
            self.producer.stop(hl2ss.StreamPort.RM_VLC_LEFTFRONT)
            self.producer.stop(hl2ss.StreamPort.RM_VLC_RIGHTFRONT)

        hl2ss.stop_subsystem_pv(HOST_IP, hl2ss.StreamPort.PERSONAL_VIDEO)

        #reset variables
        self.sink_pv = None
        self.sink_depth = None
        
        self.sink_si = None
        self.sink_vlclf = None
        self.sink_vlcrf = None

        #update class members
        self.running = False

    def isRunning(self):
        return self.running
    

    # TAKING DATA from stream
    # ALWAYS CHECKS IF STREAM IS RUNNING

    def waitReady(self):
        if self.isRunning():
            self.sink_depth.acquire()

    def getData(self):
        if self.isRunning():
            #get depth and rgb data from appropriate sensors
            #NOTE: we align rgb to longthrow instead of vice versa because longthrow is slower.
            #so we can't get a best longthrow for a given rgb, but we can get a best rgb for a given longthrow
            _, data_lt = self.sink_depth.get_most_recent_frame()
            if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                return None

            _, data_pv = self.sink_pv.get_nearest(data_lt.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                return None
            
            if self.spatial_input_enable:
                _, data_si = self.sink_si.get_nearest(data_lt.timestamp)
                if (data_si is None):
                    return None
            else:
                data_si = None

            if self.vlc_front_enable:
                _, data_lf = self.sink_vlclf.get_nearest(data_lt.timestamp)
                _, data_rf = self.sink_vlcrf.get_nearest(data_lt.timestamp)
                
                if (data_lf is None or data_rf is None):
                    return None
                
                # this is how you keep 
                # data_lf.payload = data_lf.payload.T[:,::-1]
                # data_rf.payload = data_rf.payload.T[::-1,:]

            else:
                data_lf = None
                data_rf = None
            
            #TODO: add more data on here as we think about more things

            #PV uses autofocus (intrinsics change), so we must store intrinsics in our data packet
            #save changes to extrinsics directly to member
            self.pv_intrinsics = hl2ss.update_pv_intrinsics(self.pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
            color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(self.pv_intrinsics, self.pv_extrinsics)

            return Hl2ssData(data_lt, data_pv, color_intrinsics, color_extrinsics, data_si, data_lf, data_rf)
        else:
            return None
