import os
import argparse
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
#import importlib.util #模組引入檢查 

parser = argparse.ArgumentParser()
parser.add_argument('--graph', help='Name of the .tflite file', required=True)
parser.add_argument('--label', help='Name of the label.txt file', required=True)
parser.add_argument('--threshold', help='Minimum threshold for displaying the bbox', default=0.7)
parser.add_argument('--video', help='Path of the video file', required=True)
parser.add_argument('--outputvideo', help='Path of the output video', required=True)

args = parser.parse_args()

GRAPH = args.graph
LABEL_MAP = args.label
min_confidence_threshold = float(args.threshold)
VIDEO_PATH = args.video
OUTPUT_VIDEO_PATH = args.outputvideo

CWD_PATH = os.getcwd()################可改
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_PATH)
GRAPH_PATH = os.path.join(CWD_PATH,GRAPH)
LABEL_PATH = os.path.join(CWD_PATH,LABEL_MAP)

with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()] #strip() 去除左右空格
    
#Load model
interpreter = Interpreter(model_path=GRAPH_PATH)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

#Output Video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (int(imW), int(imH)))
#out = cv2.VideoWriter(OUTPUT_NAME, fourcc, 20.0, (640, 320))

while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_confidence_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # All the results have been drawn on the frame, so it's time to display it.
    out.write(frame)
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
out.release()
cv2.destroyAllWindows()






