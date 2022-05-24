import argparse
import os

LABEL_PATH = "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)

class MyParser:

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(description='--- Raspbery Pi Urban Mobility Tracker ---')
        parser.add_argument('-modelpath', dest='model_path', type=str, required=True, help='specify path of a custom detection model')
        parser.add_argument('-labelmap', dest='label_map_path', default=DEFAULT_LABEL_MAP_PATH, type=str, required=True, help='specify the label map text file')
        parser.add_argument('-imageseq', dest='image_path', type=str, required=False, help='specify an image sequence')
        parser.add_argument('-video', dest='video_path', type=str, required=False, help='specify video file')
        parser.add_argument('-camera', dest='camera', default=False, action='store_true', help='specify this when using the rpi camera as the input')
        parser.add_argument('-threshold', dest='threshold', type=float, default=0.5, required=False, help='specify a custom inference threshold')
        parser.add_argument('-nframes', dest='nframes', type=int, required=False, default=10, help='specify nunber of frames to process')
        parser.add_argument('-display', dest='live_view', required=False, default=False, action='store_true', help='add this flag to view a live display. note, that this will greatly slow down the fps rate.')
        parser.add_argument('-save', dest='save_frames', required=False, default=False, action='store_true', help='add this flag if you want to persist the image output. note, that this will greatly slow down the fps rate.')
        parser.add_argument('-placement', dest='placement', required=True, type=str, default='above', help='give parameter "above" or "facing" meaning if the camera is above the door or facing the door, to count properly')
        
        return parser.parse_args()
