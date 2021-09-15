from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import matplotlib.pyplot as plt
import keras


img_height = 244
img_width = 244

batch_size = 16

def plot_it(H):
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    loss = H.history['loss']
    val_loss = H.history['val_loss']

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(10), acc, label='acurácia treino')
    plt.plot(range(10), val_acc, label='acurácia validação')
    plt.legend()
    plt.title('Acurácias')

    plt.subplot(1, 2, 2)
    plt.plot(range(10), loss, label='loss treino')
    plt.plot(range(10), val_loss, label='loss validação')
    plt.legend()
    plt.title('Loss')

    plt.show()
    pass


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

def model_normie():
    model = Sequential ([       
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        Conv2D(16,3, padding='same',activation='relu'),

        MaxPooling2D(),
        Conv2D(32, 3, padding='same',activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same',activation='relu'),
        #MaxPooling2D(),
        Flatten(),

        Dense(128, activation='relu',input_dim=5),

        Dense(5, activation='softmax')
        ]
    )

    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return model

def nn_no_transfer(data, labels, nome ):
    #x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 42, test_size = 0.20)
    AUTOTUNE = tf.data.AUTOTUNE
    data = data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    labels = labels.cache().prefetch(buffer_size=AUTOTUNE)
    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)

    model = model_normie()

    model.summary()
    H = model.fit(data,batch_size=batch_size, epochs=10, validation_data=labels)

    plot_it(H)

    #x_test_n = model.predict(x_test)
    y_pred = model.predict(labels)
    y_pred = np.argmax(y_pred, axis=1)

    #y_true = np.concatenate([y for x, y in labels], axis=0)
    #y_true = np.argmax(y_true, axis=1)

    print("Coletando resultados para ",nome)
    #print(sklearn.metrics.classification_report(y_true, y_true,zero_division=1))

    pass

def nn_transfer_learn():
    pass

def check_gpu():
    print(device_lib.list_local_devices())
    pass