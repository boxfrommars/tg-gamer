import cv2
import mss
import os
import signal

from src import Template, MssScreenReader, ScreenDetector, State, StateHistory, Action, Writer, GameField, Gamer, \
    Game, KeyboardController


templates_config = [{
    'name': 'br_xs',
    'src': 'templates/tmpl_br_xs.png',
    'bbox': [-35, 0, 100, 50]
}, {
    'name': 'br_xl',
    'src': 'templates/tmpl_br_xl.png',
    'bbox': [-80, 0, 150, 50]
}, {
    'name': 'br_l',
    'src': 'templates/tmpl_br_l.png',
    'bbox': [-130, 0, 150, 50]
}, {
    'name': 'br_m',
    'src': 'templates/tmpl_br_m.png',
    'bbox': [-58, -5, 130, 50]
}, {
    'name': 'br_star',
    'src': 'templates/tmpl_br_star.png',
    'bbox': [-58, -5, 170, 50]
}, {
    'name': 'br_medium',
    'src': 'templates/tmpl_br_medium.png',
    'bbox': [-90, -5, 140, 50]
}, {
    'name': 'base_widerer',
    'src': 'templates/tmpl_base_widerer.png',
    'bbox': [-50, -5, 290, 40]
}, {
    'name': 'base_medium',
    'src': 'templates/tmpl_base_medium.png',
    'bbox': [-134, 0, 50, 40]
}, {
    'name': 'base_widest',
    'src': 'templates/tmpl_base_widest.png',
    'bbox': [-75, -8, 200, 40]
}, {
    'name': 'moving_medium',
    'src': 'templates/tmpl_moving_medium.png',
    'bbox': [-105, 0, 172, 60]
}, {
    'name': 'base_narrow',
    'src': 'templates/tmpl_base_narrow.png',
    'bbox': [-25, -10, 105, 40]
}, {
    'name': 'base_wide',
    'src': 'templates/tmpl_base_wide.png',
    'bbox': [-35, -10, 195, 50]
}]


class CrazyCatGamer(Gamer):
    FALL_CAT_ACTION_NAME = 'falling'

    def __init__(self, action_range=None):
        self.action_range = action_range

    def choose_actions(self, state: State, history: StateHistory) -> list:

        prev_state = history.get_state()

        actions = []
        should_cat_fall = False

        for detected_object in state.screen.objects:
            loc = detected_object.get_bbox()
            should_cat_fall = (self.action_range[0] < loc[0][0]) and (loc[0][0] < self.action_range[1])

        if should_cat_fall:  # cat should fall
            # yes, and it's not falling yet -> start falling
            if (prev_state is None) or (not prev_state.is_action_in_progress(self.FALL_CAT_ACTION_NAME)):
                actions.append((self.FALL_CAT_ACTION_NAME, Action.START))
            # yes, but it is already falling -> keep falling
            else:
                actions.append((self.FALL_CAT_ACTION_NAME, Action.CONTINUE))
        else:  # cat should not fall
            # but cat is falling -> stop it!
            if (prev_state is not None) and prev_state.is_action_in_progress(self.FALL_CAT_ACTION_NAME):
                actions.append((self.FALL_CAT_ACTION_NAME, Action.STOP))
            # else: # and cat is not falling -> do nothing
            #     pass

        return actions


def main(max_iterations=2000, video_output=False):
    path = os.path.dirname(os.path.realpath(__file__))
    templates = [
        Template(cv2.imread(os.path.join(path, cfg['src']), cv2.IMREAD_GRAYSCALE), cfg['bbox'], name=cfg['name'])
        for cfg in templates_config]

    screen_detector = ScreenDetector(templates, [1150, 1400, 520, 860], converter=cv2.COLOR_RGB2GRAY)
    gamer = CrazyCatGamer(action_range=[550, 710])
    controller = KeyboardController({
        gamer.FALL_CAT_ACTION_NAME: {Action.START: 'space'}
    })

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
    main(max_iterations=10000, video_output=True)
