from modules.extract_feat import change_ext,get_folders,dataset_download

''' globals'''
base_folder = 'dataset/dataset_updated/training_set/'
valid_folder = 'dataset/dataset_updated/validation_set/'
image_format = '.jpeg'


blocks = 2
bins = 9


def main():
    dataset_download()#comment if alredy downloadead

    change_ext(image_format,valid_folder)
    change_ext(image_format,base_folder)
    
if __name__ == '__main__':
    main()
