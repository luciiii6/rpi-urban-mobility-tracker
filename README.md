# Raspberry Pi Urban Mobility Tracker (DeepSORT + MobileNet)

## Install (Ubuntu)
First, create a new virtualenv, 
  
```sh
python3 -m venv umt
```
initialize it, 
```sh
source umt/bin/activate
```
install whell
```sh
pip3 install wheel
```
   
then install the [TensorFlow Lite runtime package](https://www.tensorflow.org/lite/guide/python) for Python:

```sh
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

Then finish with the following:
```sh
pip install git+https://github.com/paulbozans/rpi-urban-mobility-tracker
```
fix KeyError: "The name 'net/images:0' refers to a Tensor which does not exist. The operation, 'net/images', does not exist in the graph."  if neded
  
```sh
nano umt/lib/python3.8/site-packages/deep_sort_tools/generate_detections.py
```
  Change this two lines form :
  ```sh
  self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
       f"{net/input_name}:0")
  self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
       f"{net/output_name}:0")
  ```
  to this:
  ```sh
  self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
       f"{input_name}:0")
  self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
       f"{output_name}:0")
  ```
  
Lastly, test the install by running step #6 from the Raspberry Pi install instructions above.


## Usage
Since this code is configured as a cli, everything is accessible via the `umt` command on your terminal. To run while using the Raspberry Pi camera (or laptop camera) data source run the following:
``` sh
umt -camera
```
To run the tracker on an image sequence, append the `-imageseq` flag followed by a path to the images. Included in this repo are the first 300 frames from the MOT (<a target="_blank" href="https://motchallenge.net/">Multiple Object Tracking Benchmark</a>) Challenge <a target="_blank" href="https://motchallenge.net/vis/PETS09-S2L1">PETS09-S2L1</a> video. To use them, simply download/clone this repo and cd into the main directory.
```sh
umt -imageseq data/images/PETS09-S2L1/
```
To view the bounding boxes and tracking ability of the system, append the `-display` flag to view a live feed. Note that this will greatly slow down the fps and is only recommended for testing purposes.
```sh
umt -imageseq data/images/PETS09-S2L1/ -display
```
By default, only the first 10 frames will be processed. To increase or decrease this value, append the `-nframes` flag followed by an integer value.
```sh
umt -imageseq data/images/PETS09-S2L1/ -display -nframes 20
```
To persist the image frames and detections, use the `-save` flag. Saved images are then available in the `output/` directory.
```sh
umt -imageseq data/images/PETS09-S2L1/ -save -nframes 20
```
To run the tracker using a video file input, append the `-video` flag followed by a path to the video file. Included in this repo are two video clips of vehicle traffic.
```sh
umt -video data/videos/highway_01.mp4
```
In certain instances, you may want to override the default object detection threshold (default=0.5). To accompish this, append the `-threshold` flag followed by a float value in the range of [0,1]. A value closer to one will yield fewer detections with higher certainty while a value closer to zero will result in more detections with lower certainty. It's usually better to error on the side of lower certainty since these objects can always be filtered out during post processing.
```sh
umt -video data/videos/highway_01.mp4 -display -nframes 100 -threshold 0.4
```
To get the highest fps possible, append the `-tpu` flag to use the Coral USB Accelerator for inferencing.
```sh
umt -imageseq data/images/PETS09-S2L1/ -tpu
```
