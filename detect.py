import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from typing import List, Tuple
import random
import math

import cv2

# Constants
PIXEL_SIZE = 16
SEARCH_RADIUS = 100

RADAR_VALUE = 8
FIRE_VALUE = 11
FIREFIGHTER_VALUE = 20


class FirefighterMovementTracker:
    '''
    This class keeps track of telemetry data.
    '''

    def __init__(self, x0: float, y0: float):
        self.history = [(x0, y0)]
        self.__force_direction = []

    def read(self) -> Tuple[float, float, float]:
        '''
        Receives position from tracker device (emulated)
        and returns it.
        '''

        self.receive_position()
        x, y = self.history[-1]
        ang = self.get_direction()
        return x, y, ang

    def receive_position(self) -> None:
        '''
        Emulates firefighter movement. In a real-world application,
        this would fetch data from a GPS tracker.
        '''

        # Discover in what direction we are moving
        # and follow along. If there's no direction
        # (i.e. movement just started), pick a random one.
        if len(self.history) < 2:
            ang = random.random() * 2 * math.pi

        else:
            ang = self.get_direction()

        # Divert a bit from our previous direction (+/- 0.075 rad).
        delta_ang = (random.random() - 0.5) * 0.15
        ang += delta_ang

        # Move a bit in the new direction.
        x, y = self.history[-1]
        dr = random.random() * 10
        x += dr * math.cos(ang)
        y += dr * math.sin(ang)
        self.history.append((x, y))

        # We only need to store at most 2 items in history at a time.
        self.history = self.history[-2:]

    def notify(self, ang: float) -> None:
        '''
        Called when we are walking into fire. Makes our firefighter
        divert from its walking direction by 135deg. In a real-world
        application, this would send them a notification via some device
        (e.g. smartwatch).
        '''

        self.__force_direction.append((ang + 3 * math.pi / 4) % (2 * math.pi))
 
    def get_direction(self) -> float:
        '''
        Returns our current direction based on movement history.
        Returns an angle in rad between 0 and 2pi (inclusive).
        '''

        if self.__force_direction:
            return self.__force_direction.pop(0)

        assert len(self.history) >= 2
        a = self.history[-1]
        b = self.history[-2]
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        ang = math.atan2(dy, dx)
        if ang < 0:
            ang += 2 * math.pi
        return ang


class Firefighter:
    '''
    This class represents a single firefighter object.
    '''

    def __init__(self, x0: float, y0: float):
        self.tracker = FirefighterMovementTracker(x0, y0)

    def step(self, img: np.array) -> None:
        '''
        Reads position from tracker, draws firefighter icon and radar.
        Tests if we're walking into fire.
        '''

        # Read telemetry data
        x, y, ang = self.tracker.read()

        # Draw icon
        h, w = img.shape[:2]
        pad = PIXEL_SIZE//2
        x = int(max(min(x, w - pad), pad))
        y = int(max(min(y, h - pad), pad))
        img[y-pad:y+pad, x-pad:x+pad] = FIREFIGHTER_VALUE

        # Draw radar and perform test
        dx = math.cos(ang)
        dy = math.sin(ang)
        for i in range(SEARCH_RADIUS):
            _x = int(x)
            if _x >= w or _x < 0:
                break

            _y = int(y)
            if _y >= h or _y < 0:
                break

            if img[_y, _x] == FIRE_VALUE:
                print(self, 'walking into fire!!')
                self.tracker.notify(ang)

            img[_y, _x] = RADAR_VALUE
                
            x += dx
            y += dy


# Simulation parameters
# Initial pos is based on 1.jpg
fighters = []
fighters += [Firefighter(1000, 100) for i in range(4)]
fighters += [Firefighter(200, 500) for i in range(4)]


def load_fire_image(filename: str) -> np.array:
    '''
    Loads and prepares fire image.
    '''

    # Load image and apply threshold.
    img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Fill border with fire to prevent our firefighters from escaping
    # during this simulation.
    img[np.where(img > 0)] = FIRE_VALUE
    img[:, :5] = FIRE_VALUE
    img[:5, :] = FIRE_VALUE
    img[-5:, :] = FIRE_VALUE
    img[:, -5:] = FIRE_VALUE
    return img


def build_image(img_base: np.array) -> np.array:
    '''
    Builds image array from base image and objects.
    '''

    img = img_base.copy()
    for x in fighters:
        x.step(img)

    return img


def receive_fire_image() -> np.array:
    '''
    In a real-world application, receive image from drone.
    In this simulation, we use 1.jpg.
    '''
    return fire_1


# Prepare plot
fire_1 = load_fire_image('1.jpg')
fig, ax = plt.subplots(1, 1)
im = ax.imshow(fire_1, cmap=plt.hot())
plt.gca().invert_yaxis()
while True:
    img_base = receive_fire_image()
    im.set_data(build_image(img_base))
    fig.canvas.draw_idle()
    plt.pause(1/20)
