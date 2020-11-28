# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:12:23 2020

@author: adventum
"""
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from os import listdir
from os.path import isdir
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras.models import load_model
from face_detection import FaceDetection
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.facenet_model = load_model('facenet_keras.h5')
        self.face_size = (160,160)
        self.face_det = FaceDetection()
        
    # extract a single face from a given Image       
    def _face_detection(self, filename, required_size):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if type(image) is not np.ndarray:
            image = np.array(image)
        results, img = self.face_det.predict(image, 0.4)
        if not results:
            face_array = asarray(image)
        else:
            for out in results:
                x1, y1, width, height = out[0]
                if(x1<0):
                    x1=x1*-1
                try:
                    face = image[y1:height,x1:width]
                    face_img = cv2.resize(face,required_size)
                    face_array = asarray(face_img)                    
                except:
                    print("No face detected")
        return face_array
    
    # load images and extract faces for all images in a directory
    def _load_faces(self, per_dir):   
        faces = list()
        for filename in listdir(per_dir):
            path = per_dir + filename
            face = self._face_detection(path, self.face_size)
            faces.append(face)
        return faces
    
    # load a dataset that contains one subdir for each class that in turn contains images
    def _load_dataset(self, directory):
        X, y = list(), list()
        # enumerate folders, on per class
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self._load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize face dataset
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store face and labels
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)
    
    # get the face embedding for one face
    def _get_embedding(self, face_data):
        face_emb = list()
        for face_pixels in face_data:
#            print(face_pixels)
            try:
                # scale pixel values
                face_pixels = face_pixels.astype('float32')
                # standardize pixel values across channels (global)
                mean, std = face_pixels.mean(), face_pixels.std()
                face_pixels = (face_pixels - mean) / std
                # transform face into one sample
                samples = expand_dims(face_pixels, axis=0)
                # make prediction to get embedding
                yhat = self.facenet_model.predict(samples)
                face_emb.append(yhat[0])
            except:
                print('No Embeddings for the face')
        face_emb = asarray(face_emb)
        print(face_emb.shape)
        return face_emb

# develop a classifier for the Face Dataset
    def _face_classification(self, face_emb, names):
        # normalize input vectors
        print('Training Classification Model')
        in_encoder = Normalizer(norm='l2')
        face_emb = in_encoder.transform(face_emb)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(names)
        names = out_encoder.transform(names)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(face_emb, names)
        return model
    
# train the model for new faces
    def _train_model(self, directory):
        faces, names = self._load_dataset(directory)
        face_emb = self._get_embedding(faces)
        savez_compressed('frndUnk-celebrity-faces-embeddings-new.npz', face_emb, names)
        svm_model = self._face_classification(face_emb, names)
        filename = 'finalized_model_frndUnk_new.sav'
        pickle.dump(svm_model, open(filename, 'wb'))
    
# get the face embedding for one face test time
    def _get_embedding_test_face(self, face_pixels):
        try:
            # scale pixel values
            face_pixels = face_pixels.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            # transform face into one sample
            samples = expand_dims(face_pixels, axis=0)
            # make prediction to get embedding
            yhat = self.facenet_model.predict(samples)            
        except:
            print('No Embeddings for the face')
        return yhat[0]
    
# face recognition
    def predict(self, img, min_score):
        face_embedding = load('frndUnk-celebrity-faces-embeddings-new.npz') 
        svm_model = pickle.load(open('finalized_model_frndUnk_new.sav', 'rb'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if type(img) is not np.ndarray:
            img = np.array(img)
        name_list = face_embedding['arr_1']
        output = []
        try:
            face_bbox,image = self.face_det.predict(img, 0.4)
            for out in face_bbox:                
                x1, y1, width, height = out[0]
                if(x1<0):
                    x1=x1*-1
                face_img = img[y1:height,x1:width]
                face_img = cv2.resize(face_img,self.face_size)
                embedding = self._get_embedding_test_face(face_img)            
                random_face_emb = embedding
                samples = expand_dims(random_face_emb, axis=0)
                out_encoder = LabelEncoder()
                out_encoder.fit(name_list)
                yhat_class = svm_model.predict(samples)
                yhat_prob = svm_model.predict_proba(samples)
                class_index = yhat_class[0]
                class_probability = yhat_prob[0,class_index] * 100
                predict_names = out_encoder.inverse_transform(yhat_class)
                bbox = out[0]
                score = class_probability
                if score >= min_score:
                    output.append((bbox,score,predict_names[0]))
                else:
                    output.append((bbox,score,'Unknown'))
        except:
            print('No face detected')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = asarray(img)
        return img, output
