import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_render import Hl2ssRender, RenderObject
from pynput import keyboard
import time


enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    render = Hl2ssRender()
    render.start()

    pos = [0.0,0.0,0.1]
    rotation = [0, 0, 0, 1]
    scale = [0.1,0.1,0.1]
    rgba = [1,1,1,1]
    render.addPrimObject("sphere", pos, rotation, scale, rgba)

    render_objs = []
    for i in range(4):
        pos = [float(i),0.0,0.1]
        render_objs.append(RenderObject('sphere', pos,rotation,scale,rgba))
    
    render.addPrimObjects(render_objs)

    while enable:
        time.sleep(1)
    
    render.clear()
    render.stop()
    listener.join()

    print('Finishing up Server')