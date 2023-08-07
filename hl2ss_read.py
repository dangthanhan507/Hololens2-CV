#setup imports for hl2ss
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'./hl2ss/viewer'))

from hl2ss_stream import Hl2ssData

import hl2ss
import hl2ss_io

import hl2ss_3dcv

import os

import numpy as np


class Hl2ssOfflineStreamer:
    def __init__(self, path, opts):
        #storing optional feature booleans
        self.spatial_input_enable = False
        self.vlc_front_enable = False
        self.checkOptional(opts)

        self.rd_depth = hl2ss_io.create_rd(True, f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}.bin', hl2ss.ChunkSize.SINGLE_TRANSFER, None)
        self.rd_pv = hl2ss_io.sequencer(True, f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}.bin', hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24')
        self.rd_si = hl2ss_io.sequencer(True, f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.SPATIAL_INPUT)}.bin', hl2ss.ChunkSize.SPATIAL_INPUT, None)
        self.rd_lf = hl2ss_io.sequencer(True, f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.RM_VLC_LEFTFRONT)}.bin', hl2ss.ChunkSize.SINGLE_TRANSFER, None)
        self.rd_rf = hl2ss_io.sequencer(True, f'{path}/{hl2ss.get_port_name(hl2ss.StreamPort.RM_VLC_RIGHTFRONT)}.bin', hl2ss.ChunkSize.SINGLE_TRANSFER, None)

        #setup pv calibrations (placeholder)
        self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

        self.running = False

    def checkOptional(self, opts):
        if "spatial_input" in opts:
            self.spatial_input_enable = opts["spatial_input"]
        if "vlc_front" in opts:
            self.vlc_front_enable = opts['vlc_front']

    def open(self):
        self.rd_depth.open()
        self.rd_pv.open()
        self.rd_si.open()
        self.rd_lf.open()
        self.rd_rf.open()
        self.running = True

    def close(self):
        self.rd_depth.close()
        self.rd_pv.close()
        self.rd_si.close()
        self.rd_lf.close()
        self.rd_rf.close()
        self.running = False

    def getData(self):
        data_lt = self.rd_depth.read()
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            return None

        data_pv = self.rd_pv.read(data_lt.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            return None
        
        if self.spatial_input_enable:
            _, data_si = self.rd_si.read(data_lt.timestamp)
            if (data_si is None):
                return None
        else:
            data_si = None

        if self.vlc_front_enable:
            data_lf = self.rd_lf.read(data_lt.timestamp)
            data_rf = self.rd_rf.read(data_lt.timestamp)
            
            if (data_lf is None or data_rf is None):
                return None
        else:
            data_lf = None
            data_rf = None

        self.pv_intrinsics = hl2ss.update_pv_intrinsics(self.pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(self.pv_intrinsics, self.pv_extrinsics)

        return Hl2ssData(data_lt, data_pv, color_intrinsics, color_extrinsics, data_si, data_lf, data_rf)