from PIL import Image
import numpy as np

kernel = np.array([[-1,-1,-1], #kernel
                    [-1,9,-1],
                    [-1,-1,-1]])

blocks = 2
bins = 9

def pre_process(arq):
   #print(f'{i}->{arq}')
   im = cv2.imread(arq,0)
   #im = cv2.resize(im, (256,256))
   #im = cv2.GaussianBlur(im, (3, 3), 0)#3
   im = cv2.filter2D(im,-1,kernel)#nitidez
 
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
      hist  = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
      #hist  = cv2.calcHist([img],[0],None,[256],[0,256])
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
      img = pre_process(imagePath)
      if (type(img) != type(None)) and img.all != "0":
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture = imagePath[imagePath.rfind("/") + 1:].split("_")[0]
      
        # extract Haralick texture features in 4 directions extrair a textura Haralick em 4 direçoes
        features = mahotas.features.haralick(img).mean(axis=0)
      
        # update de data e labels
        data.append(features)
        labels.append(label)
  return data,labels

def lbp_it(target,label):
  data = []
  labels = []
  dataset = glob.glob(target + "/*" + image_format) 
  for arq in dataset:
    #img = Image.open(arq)
    img = pre_process(arq)
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

#-----
def augmentation():
  import glob
  import numpy as np
  import pandas as pd
  import os
  import shutil 
  import matplotlib.pyplot as plt
  from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
  %matplotlib inline

  IMG_SIZE = 255
  batch_size = 20
  import tensorflow as tf
  from tensorflow.keras.layers import Dense, Flatten, Conv2D
  from tensorflow.keras import Model


  import tensorflow_datasets as tfds 

  def augment(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label

  (train_ds, val_ds, test_ds), metadata = tfds.load(
      'tf_flowers',
       split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
       with_info=True,
       as_supervised=True,)

  #train_ds = train_ds.shuffle(1000).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
  pass

def kerasTry():
  data_augmentation = tf.keras.Sequential([
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
     layers.experimental.preprocessing.RandomRotation(0.2)])

  image = tf.expand_dims(image, 0)
  plt.figure(figsize=(10, 10))

  for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")
  pass
