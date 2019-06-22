import numpy as np
from matplotlib import pyplot as plt 
import keras
import sys
import math
import pickle
from keras.datasets import mnist
from keras.utils import to_categorical
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import optimizers
from keras.callbacks import LearningRateScheduler

import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()

(feat_train,l_train),(feat_test,l_test) = data
feat_train = feat_train.reshape(feat_train.shape[0],28,28,1)
feat_test = feat_test.reshape(feat_test.shape[0],28,28,1)
feat_train = feat_train.astype('float32')
feat_test = feat_test.astype('float32')
feat_train/=255
feat_test/=255
label_train = to_categorical(l_train,10)
label_test = to_categorical(l_test,10)
print(label_train.shape)
print(label_test.shape)

cnn_model = Sequential()

cnn_model.add(Conv2D(6, [5,5],padding='same', activation='relu', use_bias=True,input_shape=(28,28,1),kernel_initializer = 'RandomUniform'))
print(cnn_model.output_shape)

cnn_model.add(MaxPooling2D(pool_size=(2,2)))
print(cnn_model.output_shape)

cnn_model.add(Conv2D(16, [5,5],padding='valid', activation='relu', use_bias=True))
print(cnn_model.output_shape)

cnn_model.add(MaxPooling2D(pool_size=(2,2)))
print(cnn_model.output_shape)

cnn_model.add(Flatten())
cnn_model.add(Dense(120, activation='relu', use_bias=True))
cnn_model.add(Dropout(0.2))
print(cnn_model.output_shape)
cnn_model.add(Dense(80, activation='relu', use_bias=True))
print(cnn_model.output_shape)
cnn_model.add(Dense(10, activation='softmax'))
print(cnn_model.output_shape)

#lr *= (1. / (1. + self.decay * self.iterations))
learning_rate = 0.05
decay_rate = learning_rate / 100
momentum = 0.8
#
sgd = optimizers.SGD(lr = learning_rate,momentum = momentum,decay=decay_rate)
cnn_model.compile(loss='categorical_crossentropy',optimizer= sgd,metrics=['accuracy'])
history = cnn_model.fit(feat_train,label_train,epochs = 100,batch_size=128, verbose = 2,validation_data = (feat_test,label_test))
performance = cnn_model.evaluate(feat_test,label_test,verbose = 0)
print('Test Loss:', performance[0])
print('Test Accuracy: ', performance[1])
performance = cnn_model.evaluate(feat_train,label_train,verbose = 0)
print('Train Loss:', performance[0])
print('Train Accuracy: ', performance[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
