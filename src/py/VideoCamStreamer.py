from threading import Thread
import datetime
import numpy as np
import cv2
import time
import Predictor as P
import json
import math
from queue import Queue
from utils.generic import getWithDefault, getMillis

#weights = "Z8ssd7_epoch-55_loss-0.3640_val_loss-0.5617.h5"

class Cam:
    def __init__ (self,name,videoURL,outQ,fps = 2,withImg=False,skip_millis=0):
        self.name=name
        self.videoURL = videoURL
        self.outQ = outQ
        self.thread = None
        self.running = False
        self.status="stopped"
        self.fps = fps
        self.withImg = withImg
        self.skip_millis=skip_millis


    def start(self):
        print("Start reading  " + str(self.name))
        self.thread = Thread(name=self.name ,target=self._run, args=())
        self.thread.daemon=True
        self.status="runnung"
        self.running = True
        self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join(1)
        return self

    def status(self):
        if not self.running and self.thread==None:
            return "stopped"
        elif not self.running:
            return "stopping"
        else:
            return "running"

    def _run(self):
        video = cv2.VideoCapture(self.videoURL)
        video.set(cv2.CAP_PROP_POS_MSEC,self.skip_millis)
        fps     = int(video.get(cv2.CAP_PROP_FPS))
        
        if(self.fps > fps):
            every =1
        else:
           every = int(fps / self.fps) 
        print (self.videoURL+" FRAMES POR SEGUNDO " + str(fps) + "  Every " + str(every) + " selfFPS " + str(self.fps))
        count =0
        while self.running:
           
            ret, img = video.read()
            #strDateTime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #print("Camera: {}, read ret: {}, shape: ...".format(self.name,ret))
            #print("frame reading {}".format(count))
            if ret:
                count += 1
                
                if count % every != 0:
                    continue
                #print("puting element ..." + str(count))
                self.outQ.put((self.name,img))
            # else:
            #     delayMiliSeconds =  1000
            #     delaySeconds=delayMiliSeconds/1000
            #     #print("ERROR AL LEER EL VIDEO " + str(count))
            #     time.sleep(delaySeconds)
        self.running = False
        self.thread = None
        video.release()


def buildCameras(queues,camerasConf):
    cameras = {}
    for conf in camerasConf:
        if getWithDefault(conf,"activate",1):
            camera = Cam(conf["name"],conf["videoURL"],queues[conf["imageQ"]])
            cameras[conf["name"]] = camera
            camera.start()

    return cameras


#predictor = [
#   P.SSD7Predictor("placas",
#                   'weights/Z8ssd7_epoch-64_loss-0.3654_val_loss-0.5469.h5',
#                   class_threshold={1:90,
#                                    2:90,
#                                    3:90,
#                                    4:75,
#                                    5:15}
#                ),
#    P.SSD7PredictorWithPlate(
#        "placas",
#        'weights/Z8ssd7_epoch-64_loss-0.3654_val_loss-0.5469.h5',
#        class_threshold={1:90,
#                         2:90,
#                         3:90,
#                         4:75,
#                         5:35}
#                 )
#]
