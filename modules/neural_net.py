from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
import tensorflow as tf



def augmentation():
    IMG_SIZE = 255
    batch_size = 20


def augment(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    return image, label


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

def neural_net_pure():
    pass

def check_gpu():
    print(device_lib.list_local_devices())
    pass