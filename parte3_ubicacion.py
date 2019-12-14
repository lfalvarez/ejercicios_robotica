#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose. 
"""

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from apriltag import Detector
from pyglet.window import key
from PIL import Image
import cv2

import logging

logging.basicConfig()
logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem2')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()


X = 3.5
y = 5.9
angle = (np.pi/2)
height = 0.18

def process_april_tag(pose):
    print(pose)


if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_reward = 0

while True:

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    ###### Start changing the code here.
    # TODO: Decide how to calculate the speed and direction.

    k_p = 10
    k_d = 1
    
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    
    speed = 0.2 # TODO: You should overwrite this value
    
    # angle of the steering wheel, which corresponds to the angular velocity in rad/s
    steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads # TODO: You should overwrite this value

    ###### No need to edit code below.

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


    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward

    env.render()

