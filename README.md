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
* New feature added in the file human_detection_ppl_counter.py : A virtual line is drawn and people crossing it are counted. Execution process is the same.
  There are 2 counter variables present which gets updated dynamically and displayed both in video and command prompt.
  This feature is basically the people counter.
  For proper tuning, you can change and set the upper and lower limits in line 233 and 236 according to the need.
  Additional dlib library is required for executing this module.
* New file created and named as human_detection_ppl_counter_new.py which will only record if more then 1 person enters. 
  Module can be fine tuned by varying bufSize.
  Only the count of intruders is displayed on the video and the actual people count will be displayed on the command prompt.
  The extension for now is .avi. This can be changed to .mp4 but some error will be displayed on the command prompt but still the video will be recorded. Changes can be made in line 248.
* If you want to apply this as a vertical line, then:
  1. Changes should be made in line 228 and 229. we used c[1] which calculates up and down motion, i,e.. y-axis. If we change to c[0] it'll correspond to x-axis and check for left and right motion.
  2. The co-ordinates of the points in line 216 must also be changed. It will result in vertical line. For eg: "cv2.line(frame, (W//2, 0), (W//2, H), (255, 0, 0), 2)".
  3. Changes must be made in line 233 and 237. Should replace H by W (for motion corresponding to x-axis) and centroid[1] must be changed to centroid[0] in both the lines also.
  4. To make changes in color line 205 and 206 must be changed. 
  	eqn = (endY - liney1) / ((liney2 - liney1) / (linex2 - linex1)) + linex1
        if endX >= eqn:
     also change line 19, 20 and 108 to
     	(line 19)  linex1, linex2 = None, None
	(line 20)  liney1, liney2 = 0, frame_height (you can set the frame height).
	(line 108) linex1, linex2 = W // 2, W // 2
  5. Below line is nothing but left side of the line and above line is on the right side.
* There is a new file named new.py which has a variable record. Its by default False. It should be set to True when biometrics gets punched.
  

