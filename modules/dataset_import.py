import os
import glob
from pathlib import Path

base_folder = 'A:/Academic/Art-Recongtion/dataset/dataset_updated/training_set/'
valid_folder = 'A:/Academic/Art-Recongtion/dataset/dataset_updated/validation_set/'
image_format = '.jpeg'

def dataset_download():
    
    os.system("pip install -q kaggle")
    os.environ['KAGGLE_USERNAME'] = "luish3nriqu3"
    os.environ['KAGGLE_KEY'] = "2b26c0224bc647ff4f39ed01a21dc931"
    os.system("kaggle datasets download -d thedownhill/art-images-drawings-painting-sculpture-engraving")

    os.system("peazip -ext2simple art-images-drawings-painting-sculpture-engraving.zip")

    os.system("move dataset/dataset_updated/training_set/drawings  dataset/dataset_updated/training_set/0")
    os.system("move dataset/dataset_updated/training_set/engraving  dataset/dataset_updated/training_set/1")
    os.system("mv dataset/dataset_updated/training_set/engraving  dataset/dataset_updated/training_set/2")
    os.system("mv dataset/dataset_updated/training_set/engraving  dataset/dataset_updated/training_set/3")
    os.system("mv dataset/dataset_updated/training_set/engraving  dataset/dataset_updated/training_set/4")

    os.system("mv dataset/dataset_updated/validation_set/drawings  dataset/dataset_updated/validation_set/0")
    os.system("mv dataset/dataset_updated/validation_set/drawings  dataset/dataset_updated/validation_set/1")
    os.system("mv dataset/dataset_updated/validation_set/drawings  dataset/dataset_updated/validation_set/2")
    os.system("mv dataset/dataset_updated/validation_set/drawings  dataset/dataset_updated/validation_set/3")
    os.system("mv dataset/dataset_updated/validation_set/drawings  dataset/dataset_updated/validation_set/4")
    os.system("rm -r dataset/dataset_updated/training_set/2/90.png")

    os.system("ls")
    pass

def get_folders(data_base):
	data_folders = []
	for name in os.listdir(data_base):
		if(os.path.isdir(data_base + name)):
			data_folders.append(name)
	#print(data_base)
	return data_folders

def change_ext(format=image_format,target=base_folder):
    class_paths = get_folders(target)
    for c in class_paths:
        imgs = glob.glob(target + c +"/*")
        for img in imgs:
            p = Path(img)
            p.rename(p.with_suffix(image_format))


