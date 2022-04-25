import os
import sqlite3
import cv2
import time
import datetime
import numpy as np
from .utils import Utils
from .myparser import MyParser
from .validator import Validator
# deep sort
from deep_sort.tracker import Tracker
from deep_sort import nn_matching


LABEL_PATH = "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt"
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
MAX_COSINE_DIST = 0.4
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0


class Application:
    def __init__(self):
        self.counter_in = 0
        self.counter_out = 0
        self.counting_dict = {}
        self.db_path = os.path.abspath(os.curdir) + "/database/counting.db"
        self.args = None
        self.connection = None
        self.cursor = None
        self.tracker = None
        self.img_generator = None
        self.labels = None
        self.interpreter = None
        self.COLORS = None

    def preinitialization(self):
        self.args = MyParser.parse()
        Validator.validate_args(self.args)

        if os.path.isfile(self.db_path):
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
        else:
            Utils.create_database_directory(os.path.abspath(os.curdir))
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            self.cursor.execute(
                "CREATE TABLE timestamp (id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, date TEXT, direction INTEGER)")
            self.connection.commit()

        # parse label map
        self.labels = Utils.parse_label_map(self.args, DEFAULT_LABEL_MAP_PATH)
        # initialize detector
        self.interpreter = Utils.initialize_detector(self.args)
        # create output directory
        if not os.path.exists(
                'output') and self.args.save_frames:
            os.makedirs('output')
        # initialize deep sort tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", MAX_COSINE_DIST, NN_BUDGET)
        self.tracker = Tracker(metric)

        # initialize image source
        self.img_generator = Utils.initialize_img_source(self.args)

        # initialize plot colors (if necessary)
        if self.args.live_view or self.args.save_frames:
            self.COLORS = (np.random.rand(32, 3) * 255).astype(int)

    def run(self):
        for i, pil_img in enumerate(self.img_generator(self.args)):
            print('> FRAME:', i)

            # add header to trajectory file
            if i == 0:
                w, h = pil_img.size
                if self.args.placement == "facing":
                    coord_count = h/4
                else:
                    coord_count = round(h/1.3)

            # get detections
            detections = Utils.generate_detections(
                pil_img, self.interpreter, self.args.threshold)

            # proceed to updating state
            if len(detections) == 0:
                print('   > no detections...')
            else:

                # update tracker
                self.tracker.predict()
                self.tracker.update(detections)
                # print(len(detections))
                ids = []
                for track in self.tracker.tracks:
                    ids.append(track.track_id)

                for key in list(self.counting_dict):
                    if key not in ids:
                        self.counting_dict.pop(key, None)

                for track in self.tracker.tracks:
                    bbox = track.to_tlbr()
                    if track.track_id not in self.counting_dict:
                        self.counting_dict[track.track_id] = [
                            (bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2]
                    else:
                        if self.counting_dict[track.track_id][1] < coord_count and (bbox[1]+bbox[3])/2 > coord_count:
                            ls = str(datetime.now()).split()
                            self.cursor.execute(
                                f'INSERT INTO timestamp(time,date,direction) VALUES ("{ls[1]}","{ls[0]}",1);')
                            if self.args.placement == "facing":
                                self.counter_in = self.counter_in + 1
                            else:
                                self.counter_out = self.counter_out + 1
                        if self.counting_dict[track.track_id][1] > coord_count and (bbox[1]+bbox[3])/2 < coord_count:
                            ls = str(datetime.now()).split()
                            self.cursor.execute(
                                f'INSERT INTO timestamp(time,date,direction) VALUES ("{ls[1]}","{ls[0]}",0);')
                            if self.args.placement == "facing":
                                self.counter_out = self.counter_out + 1
                            else:
                                self.counter_in = self.counter_in + 1

                        self.counting_dict[track.track_id] = [
                            (bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2]
                self.connection.commit()

            # only for live display
            if self.args.live_view or self.args.save_frames:

                # convert pil image to cv2
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                cv2.line(cv2_img, (int(50), int(coord_count)), (int(
                    pil_img.width - 50), int(coord_count)), (55, 55, 55), 5)

                # cycle through actively tracked objects
                for track in self.tracker.tracks:
                    bbox = track.to_tlbr()
                    class_name = self.labels[0]
                    color = self.COLORS[int(track.track_id) %
                                        len(self.COLORS)].tolist()
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(
                        bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                        len(str(class_name))+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(cv2_img, str(class_name) + "-" + str(track.track_id), (int(
                        bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                # live view
                if self.args.live_view:
                    cv2.imshow("tracker output", cv2_img)
                    cv2.waitKey(1)

                # persist frames
                if self.args.save_frames:
                    cv2.imwrite(f'output/frame_{i}.jpg', cv2_img)

        cv2.destroyAllWindows()
        pass
