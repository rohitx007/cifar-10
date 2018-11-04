# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

def lr_schedule(epoch):
           
    return 0.001

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#z-score
mean =  np.mean(x_train,axis=(0,1,2,3))
std_div = np.std(x_train,axis=(0,1,2,3))

xm=x_train-mean
xtm=x_test-mean
x_train = (xm*1.0)/(std_div)
x_test = (xtm*1.0)/(std_div)

num_classes = 10

y_test = tensorflow.keras.utils.to_categorical(y_test,10)
y_train = tensorflow.keras.utils.to_categorical(y_train,10)

weight_decay = 0.0001

nn = tf.keras.models.Sequential()

nn.add(Conv2D(32, (5,5), padding='same', kernel_regularizer=regularizers.l1(weight_decay), input_shape=x_train.shape[1:]))
nn.add(Activation('relu'))
nn.add(BatchNormalization())
nn.add(Conv2D(32, (5,5), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
nn.add(Activation('relu'))
nn.add(BatchNormalization())
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(Dropout(0.2))


nn.add(Conv2D(64, (5,5), padding='same', kernel_regularizer=regularizers.l1(weight_decay), input_shape=x_train.shape[1:]))
nn.add(Activation('relu'))
nn.add(BatchNormalization())
nn.add(Conv2D(64, (5,5), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
nn.add(Activation('relu'))
nn.add(BatchNormalization())
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(Dropout(0.3))

nn.add(Conv2D(128, (5,5), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
nn.add(Activation('relu'))
nn.add(BatchNormalization())
nn.add(Conv2D(128, (5,5), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
nn.add(Activation('relu'))
nn.add(BatchNormalization())
nn.add(MaxPooling2D(pool_size=(2,2)))
nn.add(Dropout(0.4))




nn.add(Flatten())

nn.add(Dense(num_classes, activation='softmax'))

nn.summary()

#data augmentation
datagen = ImageDataGenerator(rotation_range=10,
    width_shift_range=0.15,fill_mode="wrap",zca_epsilon=0.000001,
    height_shift_range=0.15,
    horizontal_flip=True,
    )
datagen.fit(x_train)

batch_size = 32

opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=0.000001)
nn.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
nn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=82,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])



#testing
scores = nn.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

# Confusion matrix result
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd


df_cm = pd.DataFrame(cm, range(10),range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
