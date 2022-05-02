from collections import deque
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import pyautogui
from mss.base import MSSBase
from enum import Enum
from abc import ABC, abstractmethod
import time


class Template:
    def __init__(self, img: npt.NDArray, shift: list, name: str = ''):
        self.img = img
        self.shift = shift
        self.name = name


class DetectedObject:
    def __init__(self, bbox, template: Template):
        self.bbox = bbox
        self.template = template

    def get_bbox(self):
        return self.bbox

    def __repr__(self):
        bbox = self.get_bbox()

        return f"{self.template.name}: {bbox[0]}, {bbox[1]}"


class MssScreenReader:
    def __init__(self, sct: MSSBase, monitor: int = 1, bbox: list = None):
        self.sct = sct
        self.monitor = monitor
        self.bbox = bbox

    def read(self):
        """Get BGR screenshot"""
        img = np.array(self.sct.grab(self.sct.monitors[self.monitor]))

        if self.bbox:
            img = img[
                  self.bbox[0]:self.bbox[1],
                  self.bbox[2]:self.bbox[3]]

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img


class Screen:
    def __init__(self, objects: List[DetectedObject], img):
        self.objects = objects
        self.img = img


class ScreenDetector:
    def __init__(self, templates: List[Template], bbox: [] = None, merge_radius=50, converter=None):
        self.templates = templates
        self.bbox = bbox
        self.merge_radius = merge_radius
        self.converter = converter

    def detect(self, img) -> Screen:
        locs = []
        if self.converter is not None:
            img_cvt = cv2.cvtColor(img, self.converter)
        else:
            img_cvt = img

        for template in self.templates:
            locs += self.detect_template(img_cvt, template)

        return Screen(objects=locs, img=img)

    def detect_template(self, img, template):
        if self.bbox is not None:
            img = img[self.bbox[0]:self.bbox[1], self.bbox[2]:self.bbox[3]]
            top_left = (self.bbox[2], self.bbox[0])
        else:
            top_left = (0, 0)

        res = cv2.matchTemplate(img, template.img, cv2.TM_CCOEFF_NORMED)
        sensitivity = 0.995  # @TODO who responsible for sensitivity?

        locs = np.where(res >= sensitivity)

        res = []

        for pt in zip(*locs[::-1]):
            detected_bbox = [
                [pt[0] + template.shift[0] + top_left[0], pt[1] + template.shift[1] + top_left[1]],
                [pt[0] + template.shift[2] + top_left[0], pt[1] + template.shift[3] + top_left[1]]
            ]

            candidate_object = DetectedObject(bbox=detected_bbox, template=template)

            if not self.is_same_object(candidate_object, res):
                res.append(candidate_object)

        return res

    def is_same_object(self, candidate: DetectedObject, existed: List[DetectedObject]) -> bool:
        """Check is object the same as alreadey detected"""

        for existed_object in existed:
            # manhattan distance between objects
            existed_bbox = existed_object.get_bbox()
            candidate_bbox = candidate.get_bbox()

            distance = abs(existed_bbox[0][0] - candidate_bbox[0][0]) + abs(existed_bbox[0][1] - candidate_bbox[0][1])

            # if distance less then merge radius then object to check is the same object as existing object
            if distance < self.merge_radius:
                return True

        return False


class State:
    def __init__(self, screen: Screen, current_actions: set = None):
        self.screen = screen
        if current_actions is None:
            self.current_actions = set()
        else:
            self.current_actions = current_actions.copy()

    def is_action_in_progress(self, action_name):
        return action_name in self.current_actions

    def set_action(self, action_name):
        self.current_actions.add(action_name)

    def remove_action(self, action_name):
        self.current_actions.remove(action_name)

    def process(self, action):
        action_name, what_to_do = action

        if what_to_do in {Action.START, Action.CONTINUE}:
            self.set_action(action_name)


class StateHistory:
    def __init__(self, max_size=1):
        self.max_size = max_size

        self.states = deque(maxlen=max_size)

    def add_state(self, state):
        self.states.append(state)

    def get_state(self, index=-1) -> State | None:
        if len(self.states):
            return self.states[index]
        else:
            return None


class GameField:
    def __init__(self, reader, detector):
        self.reader = reader
        self.detector = detector

    def get_state(self):
        img = self.reader.read()
        screen = self.detector.detect(img)
        current_state = State(screen=screen)

        return current_state


def apply_template_boundaries(img, detected_objects):
    clr = (255, 0, 255)
    for detected_object in detected_objects:
        bbox = detected_object.get_bbox()
        img = cv2.rectangle(img, bbox[0], bbox[1], clr, 4)
        img = cv2.putText(img, f'{detected_object.template.name}: {bbox[0][0]}',
                          (bbox[0][0], bbox[0][1] - 20),
                          cv2.FONT_HERSHEY_PLAIN, 3, clr, 2)
    return img


class Action(Enum):
    START = 'start'
    STOP = 'stop'
    CONTINUE = 'continue'


class Gamer(ABC):
    @abstractmethod
    def choose_actions(self, state: State, history: StateHistory) -> list:
        pass


class Writer:
    def __init__(self, writer, gamer, detector):
        self.writer = writer
        self.gamer = gamer
        self.detector = detector

    def write(self, state, actions: List[tuple]):
        detector_area_color = (0, 255, 0)
        action_range_color = (0, 0, 255)
        action_string_color = (255, 0, 0)

        img = state.screen.img

        img = apply_template_boundaries(img, state.screen.objects)

        img = cv2.rectangle(
            img,
            [self.detector.bbox[2], self.detector.bbox[0]], [self.detector.bbox[3], self.detector.bbox[1]],
            detector_area_color, 4)

        img = cv2.line(img, [self.gamer.action_range[0], 0], [self.gamer.action_range[0], 1480], action_range_color)
        img = cv2.line(img, [self.gamer.action_range[1], 0], [self.gamer.action_range[1], 1480], action_range_color)

        actions_string = ' | '.join([a[0] + '' + str(a[1]) for a in actions])
        img = cv2.putText(img, actions_string, (90, 90), cv2.FONT_HERSHEY_PLAIN, 4, action_string_color, 3)

        self.writer.write(img)

    def release(self):
        self.writer.release()


class Controller(ABC):
    @abstractmethod
    def process(self, action):
        pass


class KeyboardController(Controller):
    def __init__(self, action_map: dict = None):
        self.action_map = {} if action_map is None else action_map

    def process(self, action):
        action_name, what_to_do = action
        key = self.action_map.get(action_name, {}).get(what_to_do)

        if key is not None:
            pyautogui.press(key)


class Game:
    def __init__(self, field: GameField, gamer: Gamer, controller: Controller, writer: Writer = None):
        self.field = field
        self.gamer = gamer
        self.controller = controller

        self.writer = writer

        self.state_history = StateHistory()
        self.iteration = 0
        self.loop_start_time = None
        self.fps = None
        self.is_running = True

    def loop(self):
        self.iteration += 1

        if self.loop_start_time is None:
            self.loop_start_time = time.time()

        current_state = self.field.get_state()
        actions_to_do = self.gamer.choose_actions(current_state, self.state_history)

        for action in actions_to_do:
            self.controller.process(action)
            current_state.process(action)

        self.state_history.add_state(current_state)

        if self.writer:
            self.writer.write(current_state, actions_to_do)

        self.recalc_fps()

    def stop(self):
        self.is_running = False
        self.writer.release()

    def recalc_fps(self):
        self.fps = self.iteration / (time.time() - self.loop_start_time)

    def get_info(self):
        return {'FPS': self.fps}
