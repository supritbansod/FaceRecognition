# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:59:44 2020

@author: adventum
"""

import cv2
import os
from tqdm import tqdm
from face_recognition import FaceRecognition
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--mode", required=True,
#	help="mode of face recognition, 'cam'-->webcam or 'vid'-->stored video")
#ap.add_argument("-d", "--directory", required=True,
#	help="path to root folder where person data is stored")
#ap.add_argument("-v", "--input", 
#	help="path to video input if mode is stored video")
#ap.add_argument("-o", "--output", required=True,
#	help="name of output video")
#ap.add_argument("-c", "--confidence", type=float, default=0.7,
#	help="confidence value for correct face recognition")
#args = vars(ap.parse_args())

FR = FaceRecognition()

def videoFR(path, personCount):
    count_frames = 0
    skip_frames = 3
    inputVideo = 'Friends.mp4' #args["input"]
    prob_th = 0.85 #args["confidence"]
    cap = cv2.VideoCapture(inputVideo)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    pbar = tqdm(total=total_frames, desc="[INFO] Processing video")
    while cap.isOpened():
        ret, orig_image = cap.read()
        if not ret:
            break
        if count_frames % skip_frames == 0:
            frame, output = FR._predict(orig_image, prob_th)
            for i in range(len(output)):
                x1, y1, width, height = output[i][0]
                predict_names = output[i][2]
                cv2.rectangle(frame,(x1,y1),(width,height),(255,255,0),2)
                cv2.putText(frame,predict_names,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        video.write(frame)
#        cv2.imshow('frame',frame)
        dirCount = len(os.listdir(path))
        if(dirCount>personCount):
            print('New Person is added, train the network')
            break
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        count_frames += 1
        pbar.update(1)
    while count_frames < total_frames:
        count_frames += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    return dirCount
    
def webcamFR(path, personCount):
    prob_th = 0.85 #args["confidence"]
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # set video widht
    cap.set(4, 480) # set video height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    video = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    while cap.isOpened():
        ret, orig_image = cap.read()
        if not ret:
            break
        frame, output = FR.predict(orig_image, prob_th)
        for i in range(len(output)):
            x1, y1, width, height = output[i][0]
            predict_names = output[i][2]
            cv2.rectangle(frame,(x1,y1),(width,height),(255,255,0),2)
            cv2.putText(frame,predict_names,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        video.write(frame)
        cv2.imshow('camera',frame)
        dirCount = len(os.listdir(path))
        if(dirCount>personCount):
            print('New Person is added, train the network')
            break
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break         
    
    cap.release()
    video.release()
    cv2.destroyAllWindows()    
    return dirCount


mode1 = 'cam' #args["mode"]
path = 'FriendsDataNew/' #args["directory"]
output_path = 'output_webcam_video.mp4'#args["output"]
dirCount = len(os.listdir(path))
personCount = 7 #FR._train_model(path)
if(mode1=='vid'):
    while dirCount==personCount:
        dirCount = videoFR(path,personCount)
        if(dirCount==personCount):
            break
        personCount = FR._train_model(path)
elif(mode1=='cam'):
    while dirCount==personCount:
        dirCount = webcamFR(path,personCount)
        if(dirCount==personCount):
            break
        personCount = FR._train_model(path)
