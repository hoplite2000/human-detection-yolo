import cv2
import imutils
import numpy as np
import time
import os
import sys
import dlib
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject

model_path = "yolo-coco"
min_conf = 0.3
nms_thresh = 0.3
# Change this to True if you have GPU
USE_GPU = False
frame_width = 700
linex1, linex2 = 0, frame_width
liney1, liney2 = None, None

inputfilepath = ""
outputfilepath = ""
if len(sys.argv) > 1:
    inputfilepath = sys.argv[1]
    outputfilepath = sys.argv[2]


# load the COCO class labels our YOLO model was trained on
labelspath = os.path.sep.join([model_path, "coco.names"])
LABELS = open(labelspath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightspath = os.path.sep.join([model_path, "yolov3.weights"])
configpath = os.path.sep.join([model_path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("> Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

# check if we are going to use GPU
if USE_GPU:
    # set CUDA as the preferable backend and target
    print("> Setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("> Accessing video stream...")
cam = cv2.VideoCapture(inputfilepath if inputfilepath != "" else 0)
# cam = cv2.VideoCapture('pedestrians.mp4')
writer = None
time.sleep(1)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(cam.get(prop))
    print("> {} total frames in video".format(total))
# an error occurred while trying to determine the total number of frames in the video file
except:
    print("> Could not determine # of frames in video")
    print("> No approx. completion time can be provided")
    total = -1

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker,
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    (ret, frame) = cam.read()

    # if the frame was not ret, then we have reached the end of the stream
    if not ret:
        break

    # convert the frame from BGR to RGB for dlib
    #frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=700)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        liney1, liney2 = H//2, H//2

    rects = []

    index=LABELS.index("person")

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % 5 == 0:
        # initialize our new set of object trackers
        trackers = []

        results = []

        # create a blob of the image
        # The next line is crucial for the speed of video processing
        # The size parameter which is currently (192,192) makes the change
        # Lower the value of size, higher will be the speed of video processing but there will be slight variation in accuracy
        # Higher the value of size, Lower will be the speed of video processing but accuracy will be high and perfect. But with the help of GPU, speed will be high
        # The value of size must be a multiple of 32
        # (416,416) is the suggested value.
        #blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (192, 192), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutput = net.forward(ln)
        end = time.time()

        boxes = []
        centroids = []
        confidences = []

        # loop over each of the layer outputs
        for output in layerOutput:
            #loop over each of the detections
            for detection in output:
                scores = detection[5:]
                ID = np.argmax(scores)
                confidence = scores[ID]

                if ID == index and confidence > min_conf:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-max supression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

                # dlib correlation
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            
            eqn = ((liney2 - liney1) / (linex2 - linex1)) * (endX - linex1) + liney1
            if endY <= eqn:
                color = (0, 255, 0)
                colorc = (0, 0, 255)
            else:
                color = (0, 0, 255)
                colorc = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, colorc, -1)

    else:
        # loop over the trackers
        for tracker in trackers:

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            
            eqn = ((liney2 - liney1) / (linex2 - linex1)) * (endX - linex1) + liney1
            if endY <= eqn:
                color = (0, 255, 0)
                colorc = (0, 0, 255)
            else:
                color = (0, 0, 255)
                colorc = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, colorc, -1)

    # draw a horizontal line in the center of the frame
    cv2.line(frame, (0, H // 2), (W, H // 2), (255, 0, 0), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and (H//2)-5 < centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and (H//2)+5 > centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to



    text1 = "Above line: {}".format(totalUp)
    text2 = "Below line: {}".format(totalDown)
    cv2.putText(frame, text1, (10, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, text2, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Cam", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `esc` key was pressed, break from the loop
    if key == 27:
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1

    if outputfilepath != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outputfilepath, fourcc, 25, (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("> Single frame took {:.4f} seconds".format(elap))
            print("> Estimated total time to finish: {:.4f}".format(elap * total))

    sys.stdout.flush()
    sys.stdout.write("\r> Above line: {} and Below line: {} ".format(totalUp, totalDown))
    sys.stdout.flush()

    if writer is not None:
        writer.write(frame)

cam.release()
cv2.destroyAllWindows()