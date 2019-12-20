#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import transformations as tf
from apriltag import Detector
import cv2

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-udem1-v0')
parser.add_argument('--map-name', default='udem2')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

X = 3.5
y = 5.9
angle = (np.pi/2)
tag_size = 0.18
tile_size = 0.585

def process_april_tag(pose):
    ## Aquí jugar
    ## pose es tag con respecto al robot
    ## T_a es la transformación del april tag con respecto al mapa
    T_a = tf.translation_matrix([-X * tile_size, -tag_size * 3/4, y*tile_size])
    R_a = tf.euler_matrix(0, angle, 0)
    T_m_a = tf.concatenate_matrices(T_a, R_a)
    ## Aquí dando vuelta el robot, por cuenta del cambio de los angulos
    T_r_a = np.dot(pose, tf.euler_matrix(0, np.pi, 0))
    ##print(T_r_a)
    T_a_r = np.linalg.inv(T_r_a)
    T_m_r = np.dot(T_m_a, T_a_r)
    print(T_m_r)

from PIL import Image

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    ###### No need to edit code below.

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)

        im.save('screen.png')

    tag_size = 0.18
    tile_size = 0.585
    camera = [305.57, 308.83, 303.07, 231.88]
    original = Image.fromarray(obs)

    gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    detector = Detector()
    detections, _ = detector.detect(gray, return_image=True)

    for detection in detections:
        pose, e0, e1 = detector.detection_pose(detection, camera, tag_size * tile_size)
        process_april_tag(pose)
        print('hola')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
