from modules.dataset_import import change_ext,get_folders,dataset_download
from modules.extract_feat import extract_features,forest_it
from neural_net import nn_no_transfer,nn_transfer_learn
import glob
import cv2 
import numpy as np
import pandas as pd 
import tensorflow as tf
import os


''' globals'''
base_folder = 'A:/Academic/Art-Recongtion-Copia/dataset/dataset_updated/training_set/'
valid_folder = 'A:/Academic/Art-Recongtion-Copia/dataset/dataset_updated/validation_set/'
image_format = '.jpeg'

labels=["draw","engr","icon","paint","sculp"]

blocks = 2
bins = 9

img_height = 244
img_width = 244

batch_size = 16

def main():
    #dataset_download()#comment if alredy downloadead
    i=2
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #change_ext(image_format,valid_folder)#comment if alredy changed the ext
    #change_ext(image_format,base_folder)
    '''
    while(i<3):
        i+=1
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_folder,
        label_mode='categorical', #multiclass
        image_size=(img_height, img_width),
        batch_size = batch_size,
        subset='training',
        validation_split = 0.2,
        seed=123
        )
        valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_folder,
        label_mode='categorical', #multiclass
        image_size=(img_height, img_width),
        batch_size = batch_size,
        subset='validation',
        validation_split = 0.2,
        seed=123
        )
        #dt, lb = extract_features("lbp")	
        #dhisto, lbhisto = extract_features("hist")	
        #dhara, lbhara = extract_features("hara")

        #forest_it(dhisto, lbhisto, "histograma",5)
        #forest_it(dhara, lbhara, "haralick",3)
        nn_no_transfer(train_dataset, valid_dataset, "no extract - augment")
    '''
    j=1
    while(j<3):
        j+=1
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_folder,
        label_mode='categorical', #multiclass
        image_size=(img_height, img_width),
        batch_size = batch_size,
        subset='training',
        validation_split = 0.2,
        seed=123
        )
        valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_folder,
        label_mode='categorical', #multiclass
        image_size=(img_height, img_width),
        batch_size = batch_size,
        subset='validation',
        validation_split = 0.2,
        seed=123
        )
        #dt, lb = extract_features("lbp")	
        #dhisto, lbhisto = extract_features("hist")	
        #dhara, lbhara = extract_features("hara")

        #forest_it(dhisto, lbhisto, "histograma",5)
        #forest_it(dhara, lbhara, "haralick",3)
        nn_transfer_learn(train_dataset, valid_dataset, "no extract - transfer learn")
    nn_transfer_learn(train_dataset, valid_dataset, "no extract - transfer learn")
    pass

    
    

if __name__ == '__main__':
    main()
