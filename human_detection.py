import cv2
import imutils
import numpy as np
import time
import os
import sys

model_path = "yolo-coco"
min_conf = 0.3
nms_thresh = 0.3
#Change this to True if you have GPU
USE_GPU = False
frame_width = 700
linex1 = 0
linex2 = frame_width
liney1 = 150
liney2 = 250

inputfilepath = ""
outputfilepath = ""
if len(sys.argv) > 1:
	inputfilepath = sys.argv[1]
	outputfilepath = sys.argv[2]

def detect_people(frame,net,ln,index=0):
    (H,W) = frame.shape[:2]
    results = []

    #create a blob of the image
    #The next line is crucial for the speed of video processing
    #The size parameter which is currently (192,192) makes the change
    #Lower the value of size, higher will be the speed of video processing but there will be slight variation in accuracy 
    #Higher the value of size, Lower will be the speed of video processing but accuracy will be high and perfect. But with the help of GPU, speed will be high
    #The value of size must be a multiple of 32
    #(416,416) is the suggested value.
    #blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (192,192), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutput = net.forward(ln)
    end = time.time()

    boxes = []
    centroids = []
    confidences = []

    #loop over each of the layer outputs
    for output in layerOutput:
        #loop over each of the detections
        for detection in output:
            scores = detection[5:]
            ID = np.argmax(scores)
            confidence = scores[ID]

            if ID == index and confidence > min_conf:
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    #apply non-max supression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)

    if len(idxs)>0:
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results, start, end
    
#load the COCO class labels our YOLO model was trained on
labelspath = os.path.sep.join([model_path, "coco.names"])
LABELS = open(labelspath).read().strip().split("\n")

#derive the paths to the YOLO weights and model configuration
weightspath = os.path.sep.join([model_path, "yolov3.weights"])
configpath = os.path.sep.join([model_path, "yolov3.cfg"])

#load our YOLO object detector trained on COCO dataset (80 classes)
print("> Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

# check if we are going to use GPU
if USE_GPU:
    # set CUDA as the preferable backend and target
    print("> Setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#initialize the video stream and pointer to output video file
print("> Accessing video stream...")
cam = cv2.VideoCapture(inputfilepath if inputfilepath != "" else 0)
#cam = cv2.VideoCapture('pedestrians.mp4')
writer = None
time.sleep(1)

#try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(cam.get(prop))
    print("> {} total frames in video".format(total))
#an error occurred while trying to determine the total number of frames in the video file
except:
    print("> Could not determine # of frames in video")
    print("> No approx. completion time can be provided")
    total = -1

#loop over the frames from the video stream
while True:
    (ret, frame) = cam.read()

    #if the frame was not ret, then we have reached the end of the stream
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=frame_width)
    results, start, end = detect_people(frame, net, ln, index=LABELS.index("person"))

    safe = set()
    
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid

        cv2.line(frame, (linex1,liney1), (linex2,liney2), (255,0,0), 1)
        #linear algebra
        eqn = ((liney2-liney1)/(linex2-linex1))*(endX-linex1) + liney1
        
        if endY <= eqn:
            color = (0, 255, 0)
            colorc = (0, 0, 255)
            safe.add(i)
        else:
            color = (0, 0, 255)
            colorc = (0, 255, 0)
            #safe.add(i)
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, colorc, 1)

    text = "Safe: {}".format(len(safe))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #comment out the below line if its taking too long to process the video
    cv2.imshow("Cam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    if outputfilepath != "" and writer is None:
        #initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outputfilepath , fourcc, 25, (frame.shape[1], frame.shape[0]), True)
        #some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("> Single frame took {:.4f} seconds".format(elap))
            print("> Estimated total time to finish: {:.4f}".format(elap * total))

    if writer is not None:
        writer.write(frame)

cam.release()
cv2.destroyAllWindows()