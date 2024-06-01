# dental-detection
from google.colab import drive
drive.mount('/content/drive')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
train_path = "/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/Trianing"
test_path = "/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/test"
model = Sequential()

model.add(Conv2D(128, 3, activation="relu", input_shape=(100,100,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, activation="relu"))
#model.add(MaxPooling2D())
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D())

model.add(Dropout(0.50))
model.add(Flatten())

model.add(Dense(5000, activation = "relu"))
model.add(Dense(1024, activation = "relu"))
# model.add(Dense(512,activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(2, activation = "softmax"))

model.summary()
model.compile(loss="categorical_crossentropy", optimizer = "SGD", metrics = ["accuracy"])
train_datagen = ImageDataGenerator(rescale = 1./255,
                  shear_range = 0.3,
                  horizontal_flip=True,
                  vertical_flip=False,
                  zoom_range = 0.3
                  )
test_datagen  = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(100,100),
                                                    batch_size = 2,
                                                    color_mode= "rgb",
                                                    class_mode = "categorical")
test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=(100,100),
                                                    batch_size = 2,
                                                    color_mode= "rgb",
                                                    class_mode = "categorical")
train_generator[0][0].shape
hist = model.fit_generator(generator = train_generator,
                   steps_per_epoch = 10,
                   epochs = 10,
                   validation_data = test_generator,
                   validation_steps =10)
from keras.models import load_model

model.save("teeth_dataset")
import tensorflow as tf
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_generator)
b=predictions[8]
print(b)
np.argmax(b)
from keras.models import load_model
model=load_model('teeth_dataset')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/test/caries/wc51.jpeg", target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
if result[0][0] == 1:
    prediction = "carries "
    print(prediction)
else:
    prediction = "healthy "
    print(prediction)
print(result)
catogori = ["carries", "healthy"]
import numpy as np # linear algebra
import matplotlib.pyplot as plt

def prepare(f):
    imgs=100
    import cv2
    img_array=cv2.imread(f)
    new_array=cv2.resize(img_array,(imgs,imgs))
    return new_array.reshape(-1, imgs,imgs,3)
    prep=[prepare('/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/test/no-caries/nc2.jpg')]

img = load_img("/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/test/no-caries/nc2.jpg")
plt.imshow(img)
plt.axis("on")
plt.show()

prediction=model.predict(prep)
print(catogori[int(prediction[0][0])])
print(int(prediction[0][0]))
print(prediction)
img = load_img("/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/test/caries/wc50.jpg")
plt.imshow(img)
plt.axis("on")
plt.show()

prep=[prepare('/content/drive/MyDrive/archive (2)/teeth_dataset/teeth_dataset/test/caries/wc50.jpg')]
prediction=model.predict(prep)
print(catogori[np.argmax(prediction)])
print(prediction)
