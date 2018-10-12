from queue import Queue
import time
import cv2
import json
import shutil
import os
import sys
import requests
import math
import numpy as np

import  VideoCamStreamer  as VCS
import Models as M
import Queues as Q
import PredictorsQR as P
from utils.generic import getWithDefault, getMillis

from utils.generic import getWithDefault,getMillis
from pylibdmtx.pylibdmtx import decode

def decodeQrDataMatrix(img):
    d=decode(img)	
    if len(d) >=1 :
        data= d[0].data
        dataDecoded= data.decode('utf-8')
        return dataDecoded
    else:
        return "unknown"
	

def decodeConfigJson(confiFile):
    with open(confiFile) as f:
        data = json.load(f)
    return data

def cleanResDirectory():
    if os.path.exists('camsResult'):
        shutil.rmtree('camsResult')
    os.makedirs('camsResult')

#CAUDAL_URL = "http://localhost:8070/event"
JSON_HEADERS = {"content-type":"application/json"}

def preProcessQrFrame (img,y_pred,scale=1):
    y_pred = np.array(y_pred["objects"])
    y_pred[:,2:] = y_pred[:,2:]/scale

    crop_img = img[ymin:ymax, xmin:xmax]
    return crop_img


def showPrediction(img,y_pred,scale=10):
    height, width, channels = img.shape
    img_disp = cv2.resize(img,(int(width/scale),int(height/scale)))
    colors = [(0,0,0),(255,0,0),(0,255,0),(255,255,255),(0,0,255),(255,255,255)]
    #print("y_pred:       {}".format(y_pred))
    y_pred = np.array(y_pred["objects"])
    #print("y_pred.shape: {}".format(y_pred.shape))

    if y_pred.shape != (0,):
        y_pred[:,2:] = y_pred[:,2:]/scale
        for cls,prob,xmin,ymin,xmax,ymax in y_pred:
            color = colors[cls]
            cv2.rectangle(img_disp,(xmin,ymin),(xmax,ymax),color,2)

            #procesamiento de qr
            #crop_img = img_disp[ymin:ymax, xmin:xmax]
            #dataReaded= decodeQrDataMatrix(crop_img)
            #print("******* DATA : "  + dataReaded )
            #img_disp = crop_img

    cv2.imshow("IMG",img_disp)


def main(configFile,param):
    cleanResDirectory()
    config = decodeConfigJson(configFile)

    #caudalURL = config["Caudal"]
    serviceLogURL =  "http://localhost:8080/logs"

    print("Sending events to: {}".format(serviceLogURL))

    models = M.buildModels(config["Models"])
    queues = Q.buildQueues(config["Queues"])
    predictors = P.buildPredictors(models,queues,config["Predictors"])
    print("creating cameras")
    cameras = VCS.buildCameras(queues,config["Cameras"])

    withImg = True if param=="manual" else False


    outQ = queues["qr_out"]
    lastDataMatrixSent = ""
    n=0
    while 1:
        component,cam,pred,img,ok = outQ.get()
        #print("Sacando elemento {}".format(n))
        n+=1
        objects = pred["objects"]
        dataMatrix = pred["dataMatrix"]
        plate = getWithDefault(pred,"plate","unknown")
        height, width, channels = img.shape
        if ok and (dataMatrix != "unknown")and lastDataMatrixSent != dataMatrix :
            lastDataMatrixSent = dataMatrix
            info = {
                "cameraId"   :cam,
                "aITime"     :getMillis(),
                "data"       :dataMatrix
            }

            print("Sending: {}".format(info))
            # try:
            #     requests.post(
            #         serviceLogURL,
            #         headers=JSON_HEADERS,
            #         data=json.dumps(info))
            # except:
            #     print("Unexpected error:", sys.exc_info()[0])

        if withImg:
          showPrediction(img,pred,scale=2)
          cmdK = cv2.waitKey(20000) & 0xFF
          if cmdK == ord("q"):
              print("Quiting...1")
              streamCams.stopAll()
              cv2.destroyAllWindows()
              print("Quiting...2")
              exit()

# FELIPE COMENTO
if __name__ == "__main__":
    try:
        if len(sys.argv)>2:
            param=sys.argv[2]
        else:
            param=None
        configFile = sys.argv[1]
        main(configFile,param)
        #startReading()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
