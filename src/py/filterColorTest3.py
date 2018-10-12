import cv2
import numpy as np
import colorsys

videoPath = "/home/quantum/Documents/quantum/clients/gmodelo/streamServer/VIDS/20180815_17-03-11.mp4"

#videoPath = "/home/quantum/Projects/DATASETS/zacatecas/videos/pt/videos/CamPalL1/20180927_20340101004326_20340101010711_100153.mp4"

imgPath = "/home/quantum/Pictures/Screenshot from 2018-10-10 14-01-03.png"
imgPath = "/home/quantum/Pictures/Screenshot from 2018-10-11 20-25-31.png"

imgPath = "/home/quantum/Pictures/Screenshot from 2018-10-11 21-57-50.png"

cap = cv2.VideoCapture(videoPath)

minutes = 60000*6
#cap.set(cv2.CAP_PROP_POS_MSEC,minutes)




green2 =np.uint8([[[5,255,20 ]]])



lower_red = np.array([30,150,50])

color = np.uint8([[[184,174,168 ]]]) #blanco
hsv         = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

print( hsv )
print(bgr)

dark_red  = np.uint8([[[12,22,121]]])
dark_red = cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)


print("**********LIST PIXEL COLORS **************")
frame = cv2.imread(imgPath)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

print(frame)
frameCopy =[]



def barrer(pointLower):
    i=0
    limit = 100
    p=0
    for fila in frame:
        i+=1
        for point in fila:
            
           
            a,b,c= point
            a= int(point[0])
            b= int(point[1])
            c =int(point[2])
            #a,b,c= 101,16,180
            #print("punto {} a={} b={} c= {}".format(p,a,b,c) )
            pointUpper= np.array([a,b,c])
            print("LOWER {} FILA {} UPPER: {}  ".format(pointLower, i , pointUpper))
            p +=1
            # pointLower = np.array([0,17,165])
            # pointUpper = np.array([50,40,190])

            # pointLower = np.array([80,10,165])
            # pointUpper = np.array([250,40,190])

            pointLower = np.array([80,4,20])
            pointUpper = np.array([250,40,240])



            mask = cv2.inRange(hsv, pointLower, pointUpper)
            res = cv2.bitwise_and(frame,frame, mask= mask)
            cv2.imshow('res',res)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
           
        
        # if i<= limit:
        #     print("-------Elment {}".format(i))
        #     print(point)
        #     frameCopy.append(point)

lower_white =  np.array([109,22,184])
barrer(lower_white)

print(" *************** COPY  : ********")
a = np.array(frameCopy)
#print(a)



# i=0
# while(1):
#     _, frame = cap.read()
#     #frame = cv2.imread(imgPath)

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
#     # lower_red = np.array([30,150,50])
#     # upper_red = np.array([255,255,180])

#     # lower_white = np.array([0, 0, 212])
#     # upper_white = np.array([131, 255, 255])

#     # lower_blue = np.array([110,50,50])
#     # upper_blue = np.array([130,255,255])

#     # upper_white = np.array([0,0,255])
#     # lower_white =  np.array([109,22,184])
  
    
#     # ORANGE_MIN = np.array([5, 50, 50],np.uint8)
#     # ORANGE_MAX = np.array([15, 255, 255],np.uint8)
    

  
#     pointLower = np.array([80,10,165])
#     pointUpper = np.array([250,40,190])

#     mask = cv2.inRange(hsv, pointLower, pointUpper)
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#     hue ,saturation ,value = cv2.split(hsv)
#     cv2.imshow('Saturation Image',saturation)
#     print(i)
#     i+=1

#     cv2.imshow('frame',frame)
#     # cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     # #cv2.imshow('copy',a)


#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

cv2.destroyAllWindows()
cap.release()