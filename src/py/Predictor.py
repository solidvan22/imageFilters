import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import numpy as np
import sys
from queue import Queue
from threading import Thread

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
import cv2
import subprocess
from utils.generic import getWithDefault,getMillis

class  SSD7Predictor:
    def __init__ (
            self,
            name,
            model,
            inQ,
            outQ,
            img_height = 300,
            img_width = 480,
            normalize_coords = True,
            class_threshold=None,
            confidence_thresh=0.15, #0.25,
            iou_threshold= 0.05, #0.15, #0.45,
            top_k=10
            ):

            self.name = name
            self.model = model
            self.inQ = inQ
            self.outQ = outQ
            self.img_height = img_height
            self.img_width = img_width
            self.normalize_coords = normalize_coords

            self.class_threshold=class_threshold
            self.confidence_thresh=confidence_thresh
            self.iou_threshold=iou_threshold
            self.top_k=top_k
            self.thr = None
            self.running = False
            self.state = "stopped"


    def _fix_decoded(self,y_pred_decoded):
        if self.class_threshold == None:
            return y_pred_decoded
        else:
            result = []
            for box in y_pred_decoded:
                clase = box[0]
                confidence = box[1]
                threshold = self.class_threshold[str(clase)]
                if confidence>=threshold:
                    result.append(box)
            return result

    def preProccessing(self,frm):
        self.realHeight,self.realWidth = frm.shape[:2]
        img = cv2.resize(frm,(self.img_width,self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch_images = np.expand_dims(img,axis=0)
        return batch_images

    def postProccessing(self,y_pred):
        y_pred_decoded = decode_detections(
            y_pred,
            confidence_thresh=self.confidence_thresh, #0.25,
            iou_threshold= self.iou_threshold, #0.15, #0.45,
            top_k=self.top_k, #200,
            normalize_coords=self.normalize_coords,
            img_height=self.realHeight,
            img_width=self.realWidth
        )
        y_pred_decoded = y_pred_decoded[0]
        if y_pred_decoded.shape != (0,):
            y_pred_decoded[:,1] *= 100
            y_pred_decoded = y_pred_decoded.astype(int)
            y_pred_fixed = self._fix_decoded(y_pred_decoded)
        else:
            y_pred_fixed = y_pred_decoded
        return y_pred_fixed

    def predict(self,camera,frm):
        batch_images = self.preProccessing(frm)
        with K.get_session().as_default():
            with tf.get_default_graph().as_default():
                y_pred = self.model.getModel().predict(batch_images)
        return self.postProccessing(y_pred)

    def start(self):
        print("starting "+ self.name)
        if not self.running and self.thr==None:
            print("creating thread")
            self.thr = Thread(name=self.name+".helper",target=self._run, args=())
            self.thr.daemon = True
            self.running = True
            self.thr.start()
            print("thread created")
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.thr.join(1)
        return self

    def _run(self):
        while self.running:
            camera,frm = self.inQ.get()
            result = self.predict(camera,frm)
            self.outQ.put((self.name,camera,{"objects":result},frm,True))

        self.dest_queue.put((self.name,None,None,None,False))
        self.running = False
        self.thr = None
        return self

    def status(self):
        if not self.running and self.thr==None:
            return "stopped"
        elif not self.running:
            return "stopping"
        else:
            return "running"


class  ALPRPredictor:
    def __init__ (self,name,clipsPath,inQ,outQ):
        self.name = name
        self.clipsPath = clipsPath
        self.inQ = inQ
        self.outQ = outQ
        self.thr = None
        self.running = False
        self.state = "stopped"
        self.state = {}


    def predict(self,camera,frm):
        batch_images = self.preProccessing(frm)
        with K.get_session().as_default():
            with tf.get_default_graph().as_default():
                y_pred = self.model.getModel().predict(batch_images)
        return self.postProccessing(y_pred)

    def start(self):
        print("starting ..."+ self.name)
        if not self.running and self.thr==None:
            print("creando Thread")
            self.thr = Thread(name=self.name+".helper",target=self._run, args=())
            self.thr.daemon = True
            self.running = True
            self.thr.start()
        print("started " + self.name)
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.thr.join(1)
        return self

    def _run(self):
        while self.running:
            predecesor,cam,result,frm,ok = self.inQ.get()
            camState = getWithDefault(self.state,cam,{"plates":[],"plate":"unknown"})
            self.state[cam]=camState
            print("ALPR state: {}".format(self.state[cam]))
            if ok:
                trailer_present = False
                for pred in result["objects"]:
                    if pred[0] == 2:
                        trailer_present = True
                    if pred[0] == 5 and self.state[cam]["plate"] == "unknown":
                        xmin = pred[2]
                        ymin = pred[3]
                        xmax = pred[4]
                        ymax = pred[5]
                        clipName = self.clipsPath + "/{}.jpg"
                        clipName = clipName.format(cam)
                        cv2.imwrite(clipName,frm[ymin-5:ymax+5,xmin-5:xmax+5,:])
                        response = subprocess.check_output(["alpr", "--config", "config/openalpr.conf", "-n", "1","-c","mx","-p","mx", clipName]).decode("utf-8")
                        print("arlp: {}".format(response))
                        if response != "No license plates found.\n":
                            response = response.split("\n")[1:][0].split("\t")
                            plate = response[0].replace("    - ", "")
                            confidence = response[1].split(":")[1].strip()
                            is_match = response[2].split(":")[1].strip()
                            if is_match == "1":
                                self.state[cam]["plates"].append(plate)
                                best = {}
                                print("PLATES: {}".format(self.state[cam]))
                                for p in self.state[cam]["plates"]:
                                    cnt = getWithDefault(best,p,0)
                                    best[p] = cnt + 1
                                    if best[p] > 1:
                                        self.state[cam]["plate"] = p
                                    print("BEST: {}".format(best))
                            #cv2.putText(img_disp,plate,(xmin_s,int(ymin-12/scale)),
                            #cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
                            #print("[{}] {}".format(plate,confidence))
                if not trailer_present:
                    self.state[cam]={"plate":"unknown","plates":[]}
                result["plate"] = self.state[cam]["plate"]
                print("PUTTING {}".format(result))
                self.outQ.put((self.name,cam,result,frm,True))
            else:
                self.outQ.put((self.name,predecesor,{"name":self.name,"predecesor":predecesor,"error":"Terminó"},None,False))



        self.dest_queue.put((self.name,None,None,None,False))
        self.running = False
        self.thr = None
        return self

    def status(self):
        if not self.running and self.thr==None:
            return "stopped"
        elif not self.running:
            return "stopping"
        else:
            return "running"





class  SSD7PredictorWithPlate:
    def __init__ (
            self,
            idPredictor,
            model_path,
            img_height = 300,
            img_width = 480,
            img_channels = 3,
            intensity_mean = 127.5,
            intensity_range = 127.5,
            n_classes = 5,
            scales = [0.08, 0.16, 0.32, 0.64, 0.96],
            aspect_ratios = [0.5, 1.0, 2.0],
            two_boxes_for_ar1 = True,
            steps = None,
            offsets = None,
            clip_boxes = False,
            variances = [1.0, 1.0, 1.0, 1.0],
            normalize_coords = True,
            class_threshold=None,
            confidence_thresh=0.15, #0.25,
            iou_threshold= 0.05, #0.15, #0.45,
            top_k=10):

            self.ssd_predictor = SSD7Predictor(
                idPredictor,
                model_path,
                img_height = img_height,
                img_width = img_width,
                img_channels = img_channels,
                intensity_mean = intensity_mean,
                intensity_range = intensity_range,
                n_classes = n_classes,
                scales = scales,
                aspect_ratios = aspect_ratios,
                two_boxes_for_ar1 = two_boxes_for_ar1,
                steps = steps,
                offsets = offsets,
                clip_boxes = clip_boxes,
                variances = variances,
                normalize_coords = normalize_coords,
                class_threshold=class_threshold,
                confidence_thresh=confidence_thresh, #0.25,
                iou_threshold= iou_threshold, #0.15, #0.45,
                top_k=top_k)
            self.id = str(idPredictor)
            # en el vector plates tenemos lecturas y seguimos intentando
            # hasta huntar 5 lecuras iguales
            self.plates = []
            self.plate = "unknown"



    def predict(self,camera,frm,idFrame):
        ssd_preds,height,width = self.ssd_predictor.predict(camera,frm,idFrame)
        # ver si no tenemos plate y tenemos un plate (5) en y_pred
        # si sí invocamos alpr a ver si podemos leer la placa (5 veces)
        clip_name = "data/images/clip-{}.jpg".format(camera)
        print("ssd_preds: {}".format(ssd_preds))
        trailer_present = False
        for pred in ssd_preds:
            print("TRAE: {}".format(pred[0]))
            if pred[0] == 2:
                print("SI HAY TRAILER!")
                trailer_present = True
            if pred[0] == 5 and self.plate == "unknown":
                print("TRAE PLACA: {}".format(pred))
                xmin = pred[2]
                ymin = pred[3]
                xmax = pred[4]
                ymax = pred[5]

                cv2.imwrite(clip_name,frm[ymin-5:ymax+5,xmin-5:xmax+5,:])
                for k in range(0,30):
                    cv2.imwrite(clip_name+str(k)+".jpg",frm[ymin-k:ymax+k,xmin-k:xmax+k,:])
                response = subprocess.check_output(["alpr", "--config", "config/openalpr.conf", "-n", "1","-c","mx","-p","mx", clip_name]).decode("utf-8")
                print("arlp: {}".format(response))
                if response != "No license plates found.\n":
                    response = response.split("\n")[1:][0].split("\t")
                    plate = response[0].replace("    - ", "")
                    confidence = response[1].split(":")[1].strip()
                    is_match = response[2].split(":")[1].strip()
                    if is_match == "1":
                        self.plates.append(plate)
                        best = {}
                        print("PLATES: {}".format(self.plates))
                        for p in self.plates:
                            cnt = getWithDefault(best,p,0)
                            best[p] = cnt + 1
                            if best[p] > 1:
                                self.plate = p
                            print("BEST: {}".format(best))
                    #cv2.putText(img_disp,plate,(xmin_s,int(ymin-12/scale)),
                    #cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
                    #print("[{}] {}".format(plate,confidence))
                else:
                    print(response)
        if trailer_present == False:
            print("NO HAY TRAILERRRRR")
            self.plates=[]
            self.plate = "unknown"
        return (ssd_preds,height,width,self.plate)


def buildPredictors(models,predictorsConf):
    predictors = {}
    for conf in predictorsConf:
        model = SSD7Model(conf["modelName"],conf["weightsPath"])
        models[model.getName()] = model

    return models



def buildPredictors(models,queues,predictorsConf):
    predictors = {}
    for conf in predictorsConf:
        if getWithDefault(conf,"activate",1):
            if conf["type"]=="ssd7":
                model = models[conf["model"]]
                predictor = SSD7Predictor(
                    conf["name"],
                    model,
                    queues[conf["qIn"]],
                    queues[conf["qOut"]],
                    img_height=getWithDefault(conf,"img_height",270),
                    img_width=getWithDefault(conf,"img_width",480),
                    class_threshold=getWithDefault(conf,"class_threshold",None)
                )
            else:
                predictor = ALPRPredictor(
                    conf["name"],
                    conf["clipsPath"],
                    queues[conf["qIn"]],
                    queues[conf["qOut"]]
                )
            predictors[conf["name"]] = predictor
            print("GOT THE CONTROL")
            predictor.start()

    return predictors
