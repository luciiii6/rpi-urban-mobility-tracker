import os
from time import sleep
import tflite_runtime.interpreter as tflite
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.distance import cosine
import imutils
from imutils.video import VideoStream
from deep_sort.detection import Detection
from deep_sort_tools import generate_detections as gd

w_path = os.path.join(os.path.dirname(__file__), 'deep_sort/mars-small128.pb')
encoder = gd.create_box_encoder(w_path, batch_size=1)

class Utils:
    @staticmethod
    def calculate_line_parameters(x1, y1, x2, y2):
        #y = mx + b => b = y - mx
        m = (y2 - y1) / (x2 - x1)

        b = y1 - m*x1    

        return m,b
        
    @staticmethod
    def get_point_position(x1, y1, m, b):
        
        if y1 > (m*x1 + b):
            return 'above'
        
        if y1 < (m*x1 + b):
            return 'under'
        
        return 'on'

    @staticmethod
    def create_database_directory(curr_path):
        os.mkdir(curr_path + "/database")
        return

    @staticmethod    
    def camera_frame_generator(args):

        vs = VideoStream(src=0).start()
        sleep(1.0)
        while True:
            frame = vs.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame)

    @staticmethod
    def video_frame_generator(args):
        
        counter = 0
        cap = cv2.VideoCapture(args.video_path)
        while(cap.isOpened()):
            counter += 1
            if counter > args.nframes: break

            if cv2.waitKey(1) & 0xFF == ord('q'): break

            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame)

    @staticmethod
    def initialize_img_source(args):

        if args.video_path: return Utils.video_frame_generator
            
        if args.camera: return Utils.camera_frame_generator

    @staticmethod
    def initialize_detector(args):
     
        if args.model_path:
            model_path = args.model_path

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()


        return interpreter
        
    @staticmethod
    def generate_detections(pil_img_obj, interpreter, threshold):
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # resize image to match model input dimensions
        img = pil_img_obj.resize((input_details[0]['shape'][2], 
                                input_details[0]['shape'][1]))

        input_data = np.expand_dims(img, axis=0)
        del img
        
        # check the type of the input tensor
        if input_details[0]['dtype'] == np.float32:
            input_mean = 127.5
            input_std = 127.5
            input_data = (np.float32(input_data) - input_mean)/ input_std  
            
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        confidences = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        bboxes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
        num = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
        classes = np.squeeze(interpreter.get_tensor(output_details[3]['index']) + 1).astype(np.int32)

        keep_idx = np.less(confidences[np.greater(confidences, threshold)], 1)
        bboxes = bboxes[:keep_idx.shape[0]][keep_idx]
        classes = classes[:keep_idx.shape[0]][keep_idx]
        confidences = confidences[:keep_idx.shape[0]][keep_idx]

        # denormalize bounding box dimensions
        if len(keep_idx) > 0:
            bboxes[:,0] = bboxes[:,0] * pil_img_obj.size[1]
            bboxes[:,1] = bboxes[:,1] * pil_img_obj.size[0]
            bboxes[:,2] = bboxes[:,2] * pil_img_obj.size[1]
            bboxes[:,3] = bboxes[:,3] * pil_img_obj.size[0]
        
        # convert bboxes from [ymin, xmin, ymax, xmax] -> [xmin, ymin, width, height]
        for box in bboxes:
            xmin = int(box[1])
            ymin = int(box[0])
            w = int(box[3]) - xmin
            h = int(box[2]) - ymin
            box[0], box[1], box[2], box[3] = xmin, ymin, w, h
        
        features = encoder(np.array(pil_img_obj), bboxes)

        # munge into deep sort detection objects
        detections = [Detection(bbox, score, feature, class_name) for bbox, score, feature, class_name in zip(bboxes, confidences, features, classes)]
        del input_details
        del output_details
        del bboxes
        del classes
        del confidences
        
        return detections

    @staticmethod
    def parse_label_map(args):

        labels = {}
        for i, row in enumerate(open(args.label_map_path)):
            labels[i] = row.replace('\n','')

        return labels
