from modules.dataset_import import change_ext,get_folders,dataset_download
from modules.extract_feat import extract_features
import glob
import cv2 
import numpy as np

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import random
from sklearn.ensemble import RandomForestClassifier
import sklearn

import mahotas

''' globals'''
base_folder = 'A:/Academic/Art-Recongtion/dataset/dataset_updated/training_set/'
valid_folder = 'A:/Academic/Art-Recongtion/dataset/dataset_updated/validation_set/'
image_format = '.jpeg'


blocks = 2
bins = 9


def main():
    #dataset_download()#comment if alredy downloadead

    change_ext(image_format,valid_folder)
    change_ext(image_format,base_folder)

'''
    dt, lb = extract_features("lbp")	
    dhisto, lbhisto = extract_features("hist")	
    dhara, lbhara = extract_features("hara")

    x_train_lbp, x_test_lbp, y_train_lbp, y_test_lbp = train_test_split(dt, lb, random_state = 42, test_size = 0.20)
    x_train_histo, x_test_histo, y_train_histo, y_test_histo = train_test_split(dhisto, lbhisto, random_state = 42, test_size = 0.20)
    x_train_hara, x_test_hara, y_train_hara, y_test_hara = train_test_split(dhara, lbhara, random_state = 42, test_size = 0.20)
	
    print("Treinando o modelo")
    model_lbp = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=random.randrange(1, 1000),bootstrap=False).fit(x_train_lbp, y_train_lbp)

    model_histo = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=random.randrange(1, 1000),bootstrap=False).fit(x_train_histo, y_train_histo)

    model_hara = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=random.randrange(1, 1000),bootstrap=False).fit(x_train_hara, y_train_hara)
    #print(f':{model_hara}')


    print("...Realizando o predict...")
    print("Tamanho do Teste - LBP")
    print(x_test_lbp.shape)
    y_get_lbp = model_lbp.predict(x_test_lbp)
    print("Tamanho do Teste - Histogram")
    print(x_test_histo.shape)
    y_get_histo = model_histo.predict(x_test_histo)
    print("Tamanho do Teste - Haralick")
    print(x_test_hara.shape)
    y_get_hara = model_hara.predict(x_test_hara)



    print("Coletando resultados - LBP")
    m_lbp = sklearn.metrics.confusion_matrix(y_test_lbp, y_get_lbp)
    print(sklearn.metrics.classification_report(y_test_lbp, y_get_lbp))	

    print("Coletando resultados - Histogram")
    m_histo = sklearn.metrics.confusion_matrix(y_test_histo, y_get_histo)
    print(sklearn.metrics.classification_report(y_test_histo, y_get_histo))	

    print("Coletando resultados - Haralick")
    m_lbp = sklearn.metrics.confusion_matrix(y_test_hara, y_get_hara)
    print(sklearn.metrics.classification_report(y_test_hara, y_get_hara))	
'''   
if __name__ == '__main__':
    main()
