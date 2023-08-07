#setup imports for hl2ss
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'./hl2ss/viewer'))

import threading

from pynput import keyboard

import os

#import the hl2ss lib modules
import hl2ss
import hl2ss_io
import hl2ss_mp
import hl2ss_utilities

import hl2ss_stream
from hl2ss_map import Hl2ssMapping
import open3d as o3d

host = hl2ss_stream.HOST_IP


folder = './offline_folder/'
os.makedirs(folder,exist_ok=True)
subfolder = f'./offline_script{len(os.listdir(folder))}/'
path = os.path.join(folder,subfolder)
os.makedirs(path,exist_ok=True)

unpack = True
#can only use one depth format at a time (ahat or longthrow)
ports = [
    hl2ss.StreamPort.RM_VLC_LEFTFRONT,
    hl2ss.StreamPort.RM_VLC_LEFTLEFT,
    hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
    hl2ss.StreamPort.RM_VLC_RIGHTRIGHT,
    hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    hl2ss.StreamPort.RM_IMU_ACCELEROMETER,
    hl2ss.StreamPort.RM_IMU_GYROSCOPE,
    hl2ss.StreamPort.RM_IMU_MAGNETOMETER,
    hl2ss.StreamPort.PERSONAL_VIDEO,
    hl2ss.StreamPort.MICROPHONE,
    hl2ss.StreamPort.SPATIAL_INPUT,
    hl2ss.StreamPort.EXTENDED_EYE_TRACKER,
    ]

# RM VLC parameters
vlc_mode    = hl2ss.StreamMode.MODE_1
vlc_profile = hl2ss_stream.VLC_PROFILE
vlc_bitrate = hl2ss_stream.VLC_BITRATE

# RM Depth Long Throw parameters
lt_mode   = hl2ss.StreamMode.MODE_1
lt_filter = hl2ss.PngFilterMode.Paeth

# RM IMU parameters
imu_mode = hl2ss.StreamMode.MODE_1

#NOTE: for PV params lets be more conservative (don't use 1920x1080 or 60fps)
# PV parameters
pv_mode      = hl2ss.StreamMode.MODE_1
pv_width     = hl2ss_stream.PV_WIDTH
pv_height    = hl2ss_stream.PV_HEIGHT
pv_framerate = hl2ss_stream.PV_FPS
pv_profile   = hl2ss_stream.PV_PROFILE
pv_bitrate   = hl2ss_stream.PV_BITRATE

# Microphone parameters
microphone_profile = hl2ss.AudioProfile.AAC_24000

# EET parameters
eet_fps = 30 # 30, 60, 90

# Maximum number of frames in buffer
buffer_elements = 300


if __name__ == '__main__':

    print('Getting 3D reconstruction first')
    mapping = Hl2ssMapping()
    mapping.observe_map()
    meshes = mapping.get_o3d_mesh()
    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)
    mapping.save_map(meshes[0], path, 'map')


    print('Setting up Streaming Software Now Stream')
    # Keyboard events ---------------------------------------------------------
    start_event = threading.Event()
    stop_event = threading.Event()

    def on_press(key):
        global start_event
        global stop_event

        if (key == keyboard.Key.space):
            start_event.set()
        elif (key == keyboard.Key.esc):
            stop_event.set()

        return not stop_event.is_set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem if PV is selected ------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start receivers ---------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure_rm_vlc(False, host, hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss.ChunkSize.RM_VLC, vlc_mode, vlc_profile, vlc_bitrate)
    producer.configure_rm_vlc(False, host, hl2ss.StreamPort.RM_VLC_LEFTLEFT, hl2ss.ChunkSize.RM_VLC, vlc_mode, vlc_profile, vlc_bitrate)
    producer.configure_rm_vlc(False, host, hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss.ChunkSize.RM_VLC, vlc_mode, vlc_profile, vlc_bitrate)
    producer.configure_rm_vlc(False, host, hl2ss.StreamPort.RM_VLC_RIGHTRIGHT, hl2ss.ChunkSize.RM_VLC, vlc_mode, vlc_profile, vlc_bitrate)
    producer.configure_rm_depth_longthrow(False, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.ChunkSize.RM_DEPTH_LONGTHROW, lt_mode, lt_filter)
    producer.configure_rm_imu(host, hl2ss.StreamPort.RM_IMU_ACCELEROMETER, hl2ss.ChunkSize.RM_IMU_ACCELEROMETER, imu_mode)
    producer.configure_rm_imu(host, hl2ss.StreamPort.RM_IMU_GYROSCOPE, hl2ss.ChunkSize.RM_IMU_GYROSCOPE, imu_mode)
    producer.configure_rm_imu(host, hl2ss.StreamPort.RM_IMU_MAGNETOMETER, hl2ss.ChunkSize.RM_IMU_MAGNETOMETER, imu_mode)
    producer.configure_pv(False, host, hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.ChunkSize.PERSONAL_VIDEO, pv_mode, pv_width, pv_height, pv_framerate, pv_profile, pv_bitrate, 'rgb24')
    producer.configure_microphone(False, host, hl2ss.StreamPort.MICROPHONE, hl2ss.ChunkSize.MICROPHONE, microphone_profile)
    producer.configure_si(host, hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.ChunkSize.SPATIAL_INPUT)
    producer.configure_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss.ChunkSize.EXTENDED_EYE_TRACKER, eet_fps)

    for port in ports:
        producer.initialize(port, buffer_elements)
        producer.start(port)

    # Wait for start signal ---------------------------------------------------
    print('Press space to start recording...')
    start_event.wait()
    print('Preparing...')

    # Start writers -----------------------------------------------------------
    filenames = {port : os.path.join(path, f'{hl2ss.get_port_name(port)}.bin') for port in ports}
    writers = {port : hl2ss_io.wr_process_producer(filenames[port], producer, port, 'hl2ss simple recorder'.encode()) for port in ports}
    
    for port in ports:
        writers[port].start()

    # Wait for stop signal ----------------------------------------------------
    print('Recording started.')
    print('Press esc to stop recording...')
    stop_event.wait()
    print('Stopping...')

    # Stop writers and receivers ----------------------------------------------
    for port in ports:
        writers[port].stop()

    for port in ports:
        writers[port].join()

    for port in ports:
        producer.stop(port)

    # Stop PV Subsystem if PV is selected -------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()

    # Quit if binaries are not to be unpacked ---------------------------------
    if (not unpack):
        quit()

    print('Unpacking binaries (may take several minutes)...')

    # Unpack encoded video streams to a single MP4 file -----------------------
    ports_to_mp4 = [
        hl2ss.StreamPort.PERSONAL_VIDEO,
        hl2ss.StreamPort.MICROPHONE,
        hl2ss.StreamPort.RM_VLC_LEFTFRONT,
        hl2ss.StreamPort.RM_VLC_LEFTLEFT,
        hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
        hl2ss.StreamPort.RM_VLC_RIGHTRIGHT   
    ]

    mp4_input_filenames = [filenames[port] for port in ports_to_mp4 if (port in ports)]
    mp4_output_filename = os.path.join(path, 'video.mp4')

    if (len(mp4_input_filenames) > 0):
        hl2ss_utilities.unpack_to_mp4(mp4_input_filenames, mp4_output_filename)

    # Unpack RM Depth Long Throw to a tar file containing Depth and AB PNGs ---
    if (hl2ss.StreamPort.RM_DEPTH_LONGTHROW in ports):
        hl2ss_utilities.unpack_to_png(filenames[hl2ss.StreamPort.RM_DEPTH_LONGTHROW], os.path.join(path, 'long_throw.tar'))

    # Unpack stream metadata and numeric payloads to csv ----------------------
    for port in ports:
        input_filename = filenames[port]
        output_filename = input_filename[:input_filename.rfind('.bin')] + '.csv'
        hl2ss_utilities.unpack_to_csv(input_filename, output_filename)