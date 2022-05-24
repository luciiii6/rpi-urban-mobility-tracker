import os
import sqlite3
import cv2
import time
import datetime
import numpy as np
from utils import Utils
from myparser import MyParser
from validator import Validator
# deep sort
from deep_sort.tracker import Tracker
from deep_sort import nn_matching


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
        # y = mx + b
        self.m = 0
        self.b = 0

        self.ids = None
        self.detections = None

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

        self.labels = Utils.parse_label_map(self.args)
        self.interpreter = Utils.initialize_detector(self.args)
        if not os.path.exists(
                'output') and self.args.save_frames:
            os.makedirs('output')

        # initialize deep sort tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", MAX_COSINE_DIST, NN_BUDGET)
        self.tracker = Tracker(metric)

        self.img_generator = Utils.initialize_img_source(self.args)
        
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
                    self.m, self.b = Utils.calculate_line_parameters(10, coord_count, w-10, coord_count)              
                else:
                    coord_count = round(h/1.3)
                    self.m, self.b = Utils.calculate_line_parameters(10, coord_count, w-10, coord_count)

            # get detections
            self.detections = Utils.generate_detections(
                pil_img, self.interpreter, self.args.threshold)

            # proceed to updating state
            if len(self.detections) == 0:
                print('   > no detections...')
            else:

                # update tracker
                self.tracker.predict()
                self.tracker.update(self.detections)
                self.ids = []
                for track in self.tracker.tracks:
                    self.ids.append(track.track_id)

                for key in list(self.counting_dict):
                    if key not in self.ids:
                        self.counting_dict.pop(key, None)

                for track in self.tracker.tracks:
                    bbox = track.to_tlbr()
                    if track.track_id not in self.counting_dict:
                        self.counting_dict[track.track_id] = [
                            (bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2, Utils.get_point_position((bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2, self.m, self.b)]
                    else:
                        current_pos = Utils.get_point_position((bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2, self.m, self.b)
                        if self.counting_dict[track.track_id][2] == "under" and current_pos == "above":
                            ls = str(datetime.datetime.now()).split()
                            self.cursor.execute(
                                f'INSERT INTO timestamp(time,date,direction) VALUES ("{ls[1]}","{ls[0]}",1);')
                            if self.args.placement == "facing":
                                self.counter_in = self.counter_in + 1
                            else:
                                self.counter_out = self.counter_out + 1
                        if self.counting_dict[track.track_id][2] == "above" and current_pos == "under":
                            ls = str(datetime.datetime.now()).split()
                            self.cursor.execute(
                                f'INSERT INTO timestamp(time,date,direction) VALUES ("{ls[1]}","{ls[0]}",0);')
                            if self.args.placement == "facing":
                                self.counter_out = self.counter_out + 1
                            else:
                                self.counter_in = self.counter_in + 1

                        self.counting_dict[track.track_id] = [
                            (bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2, current_pos]
                self.connection.commit()

                print(self.counter_in, self.counter_out)

            # only for live display
            if self.args.live_view or self.args.save_frames:

                # convert pil image to cv2
                cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                cv2.line(cv2_img, (int(50), int(coord_count)), (int(
                    pil_img.width - 50), int(coord_count)), (55, 55, 55), 5)

                # cycle through actively tracked objects
                for track in self.tracker.tracks:
                    bbox = track.to_tlbr()
                    #class_name = self.labels[track.class_name]
                    class_name = self.labels[0]
                    color = self.COLORS[int(track.track_id) %
                                        len(self.COLORS)].tolist()
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(
                        bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                        len(str(class_name))+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(cv2_img, str(class_name) + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    cv2.putText(cv2_img, "Counter_in: " + str(self.counter_in), (int(w/5), int(50)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(cv2_img, "Counter_out: " + str(self.counter_out), (int(w/1.4), int(50)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(cv2_img, "Counter_total: " + str(self.counter_in-self.counter_out), (int(w/2 - w/15), int(100)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                # live view
                if self.args.live_view:
                    cv2.imshow("tracker output", cv2_img)
                    cv2.waitKey(1)

                # persist frames
                if self.args.save_frames:
                    cv2.imwrite(f'output/frame_{i}.jpg', cv2_img)
            

        cv2.destroyAllWindows()
        pass
