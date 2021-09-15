from PIL import Image

import glob
import cv2 

import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import random
from sklearn.ensemble import RandomForestClassifier
import sklearn

import pandas as pd

from modules.dataset_import import get_folders

import mahotas

kernel = np.array([[-1,-1,-1], #kernel
                    [-1,9,-1],
                    [-1,-1,-1]])

blocks = 2
bins = 9

base_folder = 'A:/Academic/Art-Recongtion-Copia/dataset/dataset_updated/training_set/'
valid_folder = 'A:/Academic/Art-Recongtion-Copia/dataset/dataset_updated/validation_set/'
image_format = '.jpeg'


def pre_process(arq,kernel):
  #print(f'{i}->{arq}')
  im = cv2.imread(arq,0)
  #im = cv2.resize(im, (256,256))
  #im = cv2.GaussianBlur(im, (3, 3), 0)#3
  #im = cv2.filter2D(im,-1,kernel)#nitidez
  return im

def extract_features(extract_mode):
  classes_folders = get_folders(base_folder)

  data = []
  data_aux = []
  labels = []
  labels_aux = []

  database = base_folder

  for f in classes_folders:
      target = database + str(f)
      #print(target)
      if extract_mode == "lbp":
        data_aux,labels_aux = lbp_it(target,f)
        data += data_aux
        labels += labels_aux
      if extract_mode == "hist":
        data_aux,labels_aux = hist_it(target,f)
        #print(f'label={labels_aux}')
        data += data_aux
        labels += labels_aux
      if extract_mode == "hara":
        data_aux,labels_aux = haralick_it(target,f)
        data += data_aux
        labels += labels_aux

  return np.asarray(data), np.asarray(labels)

def hist_it(target,label):
  histograms = []
  labels = []
  image_names = glob.glob(target + "/*" + image_format) 
  #print(image_names) 
  for image in image_names:
    # compute the color histogram
    #print(labels)
    img = cv2.imread(image)
    if (type(img) != type(None)) and img.all != "0":
      #img = Image.fromarray(img)
      labels.append(label)
      #hist  = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
      hist  = cv2.calcHist([img],[0],None,[256],[0,256])
      # normalize the histogram
      cv2.normalize(hist, hist)
      #print(hist.flatten())
      histograms.append(hist.flatten())

  #print(labels)
  return histograms,labels

def haralick_it(target,label):
  data = []
  labels = []
  image_names = glob.glob(target + "/*"+image_format)  
  #print(image_names)
  for imagePath in image_names:
      # carregar imagem, converter para escala de cinza, e extrair textura
    img = pre_process(imagePath,kernel)
    if (type(img) != type(None)) and img.all != "0":
      #image = cv2.cvtColor(image, cv2.GRAY2GRAY)
      #texture = imagePath[imagePath.rfind("/") + 1:].split("_")[0]
      
      # extract Haralick texture features in 4 directions extrair a textura Haralick em 4 direçoes
      feat = mahotas.features.haralick(img).mean(axis=0)
      
      # update de data e labels
      data.append(feat)
      labels.append(label)
  return data,labels

def lbp_it(target,label):
  data = []
  labels = []
  dataset = glob.glob(target + "/*" + image_format) 
  for arq in dataset:
    #img = Image.open(arq)
    img = pre_process(arq,kernel)
    #print(type(None))
    if (type(img) != type(None)) and img.all != "0":
      img = Image.fromarray(img)
      #img = img.convert('L')
      img = np.array(img)

      #print(f'->\n{img}\n')
      #print(f':::{arq}\n')
      x1 = mahotas.features.lbp(img,3,5,True)
      #print(x1)
      data.append(x1)
      labels.append(label)
        
  return data,labels

def cm2df(cm, labels):
  df = pd.DataFrame()
  # rows
  for i, row_label in enumerate(labels):
    rowdata={}
    # columns
    for j, col_label in enumerate(labels): 
      rowdata[col_label]=cm[i,j]
      df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
  return df[labels]

def forest_it(data,labels,name,loops=1):
  
  i=0

  accura_store = []
  precis_store = []
  recall_store = []

  while(i<loops):
    i+=1
    x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 42, test_size = 0.20)
    
    print("Treinando o modelo")
    #model_lbp = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=random.randrange(1, 1000),bootstrap=False).fit(x_train_lbp, y_train_lbp)

    model = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=random.randrange(1, 1000),bootstrap=False).fit(x_train, y_train)

    print("Tamanho do Teste -",name)
    print(x_test.shape)
    x_test_n = model.predict(x_test)

    print("Coletando resultados ")
    cm = sklearn.metrics.confusion_matrix(y_test, x_test_n)
    #accura_store.append(sklearn.metrics.accuracy_score(,))
    #precis_store.append(sklearn.metrics.average_precision_score(y_test, x_test_n))
    #recall_store.append(sklearn.metrics.recall_score(y_test, x_test_n))
    print(sklearn.metrics.classification_report(y_test, x_test_n,zero_division=1))

  #print(accura_store,"Acuracia")
  #print(precis_store,"Precisão Média")
  #print(recall_store,"Recall")

  #df = cm2df(cm, labels)

  #print(df)
  pass 