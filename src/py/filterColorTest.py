import cv2
import numpy as np
import colorsys

videoPath = "/home/quantum/Documents/quantum/clients/gmodelo/streamServer/VIDS/20180815_17-03-11.mp4"

imgPath = "/home/quantum/Pictures/Screenshot from 2018-10-10 14-01-03.png"
cap = cv2.VideoCapture(videoPath)




green2 =np.uint8([[[5,255,20 ]]])



lower_red = np.array([30,150,50])

color = np.uint8([[[184,174,168 ]]]) #blanco
hsv         = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

print( hsv )
print(bgr)

dark_red  = np.uint8([[[12,22,121]]])
dark_red = cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)




while(1):
    _, frame = cap.read()
    frame = cv2.imread(imgPath)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
  
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    lower_white = np.array([0, 0, 212])
    upper_white = np.array([131, 255, 255])

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    upper_white = np.array([0,0,255])
    lower_white =  np.array([109,22,184])
    
    ORANGE_MIN = np.array([5, 50, 50],np.uint8)
    ORANGE_MAX = np.array([15, 255, 255],np.uint8)
    

  


    mask = cv2.inRange(hsv, lower_white, lower_white)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()