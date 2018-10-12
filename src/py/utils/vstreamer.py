import sys
import datetime
import json
import requests

import cv2
import numpy as np
from numpy.random import randint,seed
import tensorflow as tf
import websocket
import pandas as pd
import math

from threading import Thread
import time
from queue import Queue

from utils.generic import get_millis


class VideoStreamer:
    def __init__ (self,name,video_file,fps,dest_queue,max_frames=-1,first_frame=0,every=1,skip_millis=0):
        self.name = name
        self.video_file = video_file
        self.fps = fps
        self.dest_queue = dest_queue
        self.running = False
        self.thr = None
        self.max_frames = max_frames
        self.frame = 0
        self.first_frame = first_frame
        self.every = every
        self.skip_millis = skip_millis

    def start(self):
        print("starting "+ self.name)
        if not self.running and self.thr==None:
            self.thr = Thread(name=self.name+".helper",target=self._run, args=())
            self.thr.daemon = True
            self.running = True
            self.thr.start()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.thr.join(1)
        return self

    def _run(self):
        video = cv2.VideoCapture(self.video_file)
        video.set(cv2.CAP_PROP_POS_MSEC,self.skip_millis)
        t0 = get_millis()
        self.frame = 0
        while self.running and (self.max_frames<0 or self.frame < self.max_frames):
            self.frame += 1
            ret, image_np = video.read()
            if not ret:
                break
            if self.frame < self.first_frame:
                continue
            if self.frame % self.every != 0:
                continue
            #print("queue.size: {}".format(self.dest_queue.qsize()))
            self.dest_queue.put((self.name,image_np,True))
            t1 = get_millis()
            delta = (t1-t0)/1000.0
            delay = 1/self.fps - delta
            if delay>0:
                time.sleep(delay)
            t0=t1
        self.dest_queue.put((self.name,None,False))
        self.running = False
        self.thr = None
        video.release()
        return self

    def status(self):
        if not self.running and self.thr==None:
            return "stopped"
        elif not self.running:
            return "stopping"
        else:
            return "running"

    def current_frame(self):
        return self.frame

    def getName(self):
        return self.name

def one_not(streamers,status,printing=False):
    if printing:
        for streamer in streamers:
            print("streamer: {} -> {} {}".format(streamer.getName(),
                streamer.status(),
                streamer.current_frame()))
        print("----------------------------")
    for streamer in streamers:
        if streamer.status() != status:
            return True
    return False
