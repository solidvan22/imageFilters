import numpy as np 
import cv2 
im_width = 320
im_height = 240
cap = cv2.VideoCapture(0)


# The order of the colors is blue, green, red
lower_color_bounds = cv2.Scalar(100, 0, 0)
upper_color_bounds = cv2.Scalar(225,80,80)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(frame,lower_color_bounds,upper_color_bounds )
    mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    frame = frame & mask_rgb
    cv2.imshow("Video",frame)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()