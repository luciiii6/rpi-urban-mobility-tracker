import os
from time import sleep
import tflite_runtime.interpreter as tflite
import tensorflow as tf
from PIL import Image
import numpy as np
import pprint
import cv2
from scipy.spatial.distance import cosine
import imutils
from imutils.video import VideoStream
from deep_sort.detection import Detection
from deep_sort_tools import generate_detections as gd


# initialize an instance of the deep-sort tracker
w_path = os.path.join(os.path.dirname(__file__), 'deep_sort/mars-small128.pb')
encoder = gd.create_box_encoder(w_path, batch_size=1)

class Utils:
    @classmethod
    def calculate_line_parameters(point1, point2):
        pass

    @classmethod
    def create_database_directory(curr_path):
        os.mkdir(curr_path + "/database")
        return

    @classmethod    
    def camera_frame_gen(args):

        # initialize the video stream and allow the camera sensor to warmup
        print("> starting video stream...")
        vs = VideoStream(src=0).start()
        sleep(2.0)
        # loop over the frames from the video stream
        while True:
            # pull frame from video stream
            frame = vs.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame)
        pass

    @classmethod
    def image_seq_gen(args):

        # collect images to be processed
        images = []
        for item in sorted(os.listdir(args.image_path)):
            if item[-4:] == '.jpg': images.append(f'{args.image_path}{item}')
        
        # cycle through image sequence and yield a PIL img object
        for frame in range(0, args.nframes): yield Image.open(images[frame])

    @classmethod
    def video_frame_gen(args):
        
        counter = 0
        cap = cv2.VideoCapture(args.video_path)
        while(cap.isOpened()):
            counter += 1
            if counter > args.nframes: break
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # pull frame from video stream
            _, frame = cap.read()

            # array to PIL image format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield Image.fromarray(frame)

    @classmethod
    def initialize_img_source(args):

        # track objects from video file
        if args.video_path: return Utils.video_frame_gen
        
        # track objects in image sequence
        if args.image_path: return Utils.image_seq_gen
            
        # track objects from camera source
        if args.camera: return Utils.camera_frame_gen

    @classmethod
    def initialize_detector(args):

        TPU_PATH = 'models/tpu/mobilenet_ssd_v2_coco_quant/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        CPU_PATH = 'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'

        # initialize coral tpu model
        if args.tpu:
            print('   > TPU = TRUE')
            
            if args.model_path:
                model_path = args.model_path
                print('   > CUSTOM DETECTOR = TRUE')
                print(f'      > DETECTOR PATH = {model_path}')
                
            else:
                model_path = os.path.join(os.path.dirname(__file__), TPU_PATH)
                print('   > CUSTOM DETECTOR = FALSE')
            
            _, *device = model_path.split('@')
            edgetpu_shared_lib = 'libedgetpu.so.1'
            interpreter = tflite.Interpreter(
                    model_path,
                    experimental_delegates=[
                        tflite.load_delegate(edgetpu_shared_lib,
                            {'device': device[0]} if device else {})
                    ])
            interpreter.allocate_tensors()

        # initialize tflite model
        else:
            print('   > TPU = FALSE')
            
            if args.model_path:
                model_path = args.model_path
                print('   > CUSTOM DETECTOR = TRUE')
                print(f'      > DETECTOR PATH = {model_path}')
                
            else:
                print('   > CUSTOM DETECTOR = FALSE')
                model_path = os.path.join(os.path.dirname(__file__), CPU_PATH)

            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

        return interpreter

    @classmethod
    def generate_detections(pil_img_obj, interpreter, threshold):
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # resize image to match model input dimensions
        img = pil_img_obj.resize((input_details[0]['shape'][2], 
                                input_details[0]['shape'][1]))

        input_mean = 127.5
        input_std = 127.5
        input_data = np.expand_dims(img, axis=0)
        
        # check the type of the input tensor
        if input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - input_mean)/ input_std  
            
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        bboxes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
        classes = np.squeeze(interpreter.get_tensor(output_details[3]['index']) + 1).astype(np.int32)
        scores = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        num = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

        keep_idx = np.less(scores[np.greater(scores, threshold)], 1)
        bboxes  = bboxes[:keep_idx.shape[0]][keep_idx]
        classes = classes[:keep_idx.shape[0]][keep_idx]
        scores = scores[:keep_idx.shape[0]][keep_idx]
        pprint.pprint(len(scores))
        pprint.pprint(scores)

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
        detections = [Detection(bbox, score, feature, class_name) for bbox, score, feature, class_name in zip(bboxes, scores, features, classes)]
        return detections

    @classmethod
    def parse_label_map(args, DEFAULT_LABEL_MAP_PATH):
        if args.label_map_path == DEFAULT_LABEL_MAP_PATH: print('   > CUSTOM LABEL MAP = FALSE')
        else: print(f'   > CUSTOM LABEL MAP = TRUE ({args.label_map_path})')

        labels = {}
        for i, row in enumerate(open(args.label_map_path)):
            labels[i] = row.replace('\n','')
        
        pprint.pprint(labels)
        return labels
