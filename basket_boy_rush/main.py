import pyautogui

import cv2
import json
import mss
import numpy as np
import os
import signal
import time
import uuid
import math

from src import Template, MssScreenReader, ScreenDetector, State, StateHistory, Action, Writer, GameField, Gamer, \
    Game, Controller, HSVTemplate

BASKET_OBJECT_NAME = 'basket'
BALL_OBJECT_NAME = 'ball'
THROW_ACTION_NAME = 'throw'


class MouseController(Controller):
    def __init__(self):
        self.last_click_time = None

        self.id_ = uuid.uuid4()

        self.throw_history = {'ball': []}

    def process(self, action):
        top_left = (350, 100)  # x, y
        action_name, _, props = action

        if action_name == THROW_ACTION_NAME:
            current_time = time.time()

            ball_loc = props.get('ball')
            if ball_loc:
                abs_basll_loc = (
                    top_left[0] + int(ball_loc[0] / 2),
                    top_left[1] + int(ball_loc[1] / 2),
                )

                self.throw_history['ball'].append(abs_basll_loc)

            if (self.last_click_time is not None) and ((current_time - self.last_click_time) < 2.7):
                return

            basket_loc = props['basket']

            abs_basket_loc = (
                top_left[0] + int(basket_loc[0] / 2),
                top_left[1] + int(basket_loc[1] / 2),
            )

            basket_x = abs_basket_loc[0]
            basket_y = abs_basket_loc[1]

            xo = 464.25297363189236
            yo = 713.38807506372

            if basket_x > 750:
                a = 0.004

            if basket_x > 700:
                a = 0.012

            elif basket_x >= 650:
                a = 0.025

            elif basket_x >= 600:
                a = 0.04

            else:
                a = 0.05

            b = (-xo - basket_x) * a + (yo - basket_y) / (xo - basket_x)
            c = xo * basket_x * a + yo - xo * (yo - basket_y) / (xo - basket_x)

            click_x = -(0.49762931 * b / a) + 19.330978817352957
            click_y = (-0.24998402154960112 * (b * b) / a) + c + 180.89041488168095

            self.throw_history['click'] = (click_x, click_y)

            # with open(f'basket_boy_rush/throw_data/throw_data_{self.id_}.json', 'w') as outfile:
            #     json.dump(self.throw_history, outfile)

            pyautogui.click(click_x, click_y, 0.5, button='left')

            print('click', self.throw_history['click'])
            print('basket', abs_basket_loc)

            self.last_click_time = current_time


class BasketBoyRushGamer(Gamer):
    def __init__(self, action_range=None):
        self.action_range = action_range

    def choose_actions(self, state: State, history: StateHistory) -> list:
        basket = state.screen.get_object(BASKET_OBJECT_NAME)
        ball = state.screen.get_object(BALL_OBJECT_NAME)

        props = {}

        if basket is not None:
            basket_bbox = basket.get_bbox()
            basket_center = ((basket_bbox[1][0] + basket_bbox[0][0]) / 2, (basket_bbox[1][1] + basket_bbox[0][1]) / 2)

            props['basket'] = basket_center

        if ball is not None:
            ball_bbox = ball.get_bbox()
            ball_center = ((ball_bbox[1][0] + ball_bbox[0][0]) / 2, (ball_bbox[1][1] + ball_bbox[0][1]) / 2)

            props['ball'] = ball_center

        if basket is not None:
            return [(THROW_ACTION_NAME, Action.START, props)]

        return []


def main(max_iterations=2000, video_output=False):
    path = os.path.dirname(os.path.realpath(__file__))
    templates = [
        Template(cv2.imread(os.path.join(path, 'templates/basket.png'), cv2.IMREAD_COLOR),
                 [10, -5, 130, 18], name=BASKET_OBJECT_NAME, sensitivity=0.85),
        HSVTemplate(np.array([10, 20, 20]), np.array([12, 255, 255]), shift=[-25, 0, 45, 70], name=BALL_OBJECT_NAME)]

    screen_detector = ScreenDetector(templates, bbox=[0, 1200, 0, 1400], converter=None, merge_radius=100)
    gamer = BasketBoyRushGamer()
    controller = MouseController()

    if video_output:
        writer = Writer(
            cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1480, 1500), True),
            gamer, screen_detector)
    else:
        writer = None

    with mss.mss() as sct:
        screen_reader = MssScreenReader(sct, monitor=1, bbox=[200, 1700, 700, 2180])
        game_field = GameField(screen_reader, screen_detector)
        game = Game(game_field, gamer, controller, writer)

        signal.signal(signal.SIGINT, lambda *args: game.stop())

        while game.is_running:
            game.loop()
            if game.iteration > max_iterations:
                game.stop()

    print(game.get_info())


if __name__ == "__main__":
    main(max_iterations=500, video_output=True)

