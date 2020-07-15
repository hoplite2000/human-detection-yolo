# Human Detection

This module is human detection module developed using OpenCV which will detect all the humans uniquely in real time. This module can also used to detect humans in video clip.

### Execution

* You can execute the code by typing "python human_detection.py" in your terminal and this will start the module and wont save anything on your system.
* If you want to run this module on a video clip and save the output on your system, use "python human_detection.py <INPUT_FILENAME> <OUTPUT_FILENAME>" in your terminal. For example "python human_detection.py pedestrians.mp4 output.avi" (Use python3 if using linux system).

Note: 
* Test example and its output is also provided. If its taking too long to process a video, then just comment the "cv2.imshow("Cam", frame)" in the code.
* In the detect_people function, while creating a blob of the image
  blob = cv2.dnn.blobFromImage(frame, 1/255.0, (192,192), swapRB=True, crop=False). This line is crucial for the speed of video processing.
  The size parameter which is currently (192,192) makes the change.
  Lower the value of size, higher will be the speed of video processing but there will be slight variation in accuracy.
  Higher the value of size, Lower will be the speed of video processing but accuracy will be high and perfect. But with the help of GPU, speed will be high.
  The value of size must be a multiple of 32.
  (416,416) is the suggested value.
  

