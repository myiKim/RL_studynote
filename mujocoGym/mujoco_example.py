import distutils.util
import os
import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')

import mujoco

import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def write_frames(frames, frate=60):
    try:
        shape = (240, 320)
        type = 'gif'
        if type == 'mp4':
            #for guide, https://google.github.io/mediapy/mediapy.html#write_video
            with media.VideoWriter('./tmp/moving_body.mp4', shape, fps=frate) as writer:
                for frame_i in frames:
                    writer.add_image(frame_i)
        elif type == 'gif':
            with media.VideoWriter('./tmp/moving_body.gif', shape, fps=frate, codec='gif') as writer:
                for frame_i in frames:
                    writer.add_image(frame_i)

        print("writing success!")
        
    except:
        print("writing failed!")
        return False

    return True

    

def case1():
    xml = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba=".3 .5 .3 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    
    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)
    
    frames = []
    mujoco.mj_resetData(model, data)
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            # print(np.sum(pixels))
            
            frames.append(pixels)
    
    # Simulate and display video.
    # media.show_video(frames, fps=framerate)
    return write_frames(frames, 30)


def case2():
    tippe_top = """
        <mujoco model="tippe top">
          <option integrator="RK4"/>
        
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
             rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
          </asset>
        
          <worldbody>
            <geom size=".2 .2 .01" type="plane" material="grid"/>
            <light pos="0 0 .6"/>
            <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
            <body name="top" pos="0 0 .02">
              <freejoint/>
              <geom name="ball" type="sphere" size=".02" />
              <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
              <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
               contype="0" conaffinity="0" group="3"/>
            </body>
          </worldbody>
        
          <keyframe>
            <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
          </keyframe>
        </mujoco>
        """
    model = mujoco.MjModel.from_xml_string(tippe_top)
    renderer = mujoco.Renderer(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="closeup")
    media.show_image(renderer.render())

    duration = 7    # (seconds)
    framerate = 60  # (Hz)
    
    # Simulate and display video.
    frames = []
    mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0
    while data.time < duration:
      mujoco.mj_step(model, data)
      if len(frames) < data.time * framerate:
        renderer.update_scene(data, "closeup")
        pixels = renderer.render()
        frames.append(pixels)

    return write_frames(frames, framerate)

def myFirstOwn():
    myfirstown = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba=".3 .5 .3 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
model = mujoco.MjModel.from_xml_string(chaotic_pendulum)
renderer = mujoco.Renderer(model, 480, 640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="fixed")
media.show_image(renderer.render())

def renderMain():
    writelist = [case2()]
    print(writelist)
    
    

if __name__ =='__main__':
    renderMain()