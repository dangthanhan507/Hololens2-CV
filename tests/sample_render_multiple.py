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

    render.clear()

    pos = [0.0,0.0,0.1]
    rotation = [0, 0, 0, 1]
    scale = [0.1,0.1,0.1]
    rgba = [1,1,1,1]
        

    render_objs = []
    for i in range(4):
        pos = [float(i),0.0,0.1]
        render_objs.append(RenderObject('sphere', pos,rotation,scale,rgba))
    
    object_ids = render.addPrimObjects(render_objs)

    render.removePrimObject(object_ids[3])

    y = 0
    while enable:
        # y += 0.1
        for i in range(3):
            render_objs[i].pos = [i,y,0.1]
        #     render.transformObj(object_ids[i], render_objs[i])
        y += 0.1
        render.transformObjs(object_ids,render_objs)
        time.sleep(1)
    render.clear()
    render.stop()
    listener.join()

    print('Finishing up Server')