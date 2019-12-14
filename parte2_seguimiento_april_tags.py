#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse

import cv2
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from PIL import Image
from apriltag import Detector
import transformations as tf

import logging

logging.basicConfig()
logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.ERROR)


# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-udem1-v0')
parser.add_argument('--map-name', default='loop_sign_go')
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


LEFT = np.array([0.35, +1])
RIGHT = np.array([0.35, -1])
UP = np.array([0.44, 0.0])
DOWN = np.array([-0.44, 0])


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


def dist(matrix):
    return np.linalg.norm([matrix[0][3],matrix[1][3],matrix[2][3]])

def process_april_tag(pose):
    angulo = pose[0][3]
    distancia_frontal = pose[2][3]
    threshold_de_distancia_frontal = 0.7
    threshold_angle = 0.1 ## Esto es en radianes.
    movimientos = []
    if abs(angulo) > threshold_angle:
        if angulo > 0:
            movimientos.append(RIGHT)
        if angulo < 0:
            movimientos.append(LEFT)
    if distancia_frontal > threshold_de_distancia_frontal:
        movimientos.append(UP)
    return movimientos


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = UP
    if key_handler[key.DOWN]:
        action = DOWN
    if key_handler[key.LEFT]:
        action = LEFT
    if key_handler[key.RIGHT]:
        action = RIGHT
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)
        im.save('screen.png')

    ### apriltags detector ###
    tag_size = 0.18
    tile_size = 0.585
    camera = [305.57, 308.83, 303.07, 231.88]
    original = Image.fromarray(obs)
    cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    detector = Detector()
    detections, _ = detector.detect(gray, return_image=True)
    for detection in detections:
        pose, e0, e1 = detector.detection_pose(detection, camera, tag_size * tile_size)
        if not np.isnan(pose[0][0]):
            next_moves = process_april_tag(pose)
            for action in next_moves:
                env.step(action)

    ##########################
    if done:
        print('done!')
        ##env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
