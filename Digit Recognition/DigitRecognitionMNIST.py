!pip install keras-tuner
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import struct as st


filename = {'images' : '/content/drive/My Drive/Colab Notebooks/DigitRecognitionMnist/train-images.idx3-ubyte' ,'labels' : '/content/drive/My Drive/Colab Notebooks/DigitRecognitionMnist/train-labels.idx1-ubyte'}

train_imagesfile = open(filename['images'],'rb')
train_labelsfile=  open(filename['labels'],'rb')


train_imagesfile.seek(0)
magic = st.unpack('>4B',train_imagesfile.read(4))
train_labelsfile.seek(0)
magic1 = st.unpack('>4B',train_labelsfile.read(4))


nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column


cols= st.unpack('>I',train_labelsfile.read(4))[0] #num of Images


nBytesTotal1 = cols*1 #since each pixel data is 1 byte
y=  np.asarray(st.unpack('>'+'B'*nBytesTotal1,train_labelsfile.read(nBytesTotal1))).reshape((cols,1))

nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
X= 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))

#standardizing the input values
X=X/255
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=(1/12))


arr = np.array(y_train)
data=arr.flatten()
df = pd.DataFrame()
df['label'] = (data)
y_train=df['label']


X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])
X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])

from tensorflow import keras
from keras.layers import Dense,Dropout
from keras.layers import ReLU,LeakyReLU,PReLU,ELU
from keras.models import Sequential 



def bulid_model(hp):
  model=keras.Sequential([
                          keras.layers.Conv2D(filters=64,
                                              kernel_size=(3,3),
                                              strides=(1,1),
                                              activation='relu',
                                              kernel_initializer='he_uniform',
                                              input_shape=(1,28,28),
                                              data_format='channels_first'),
                          keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
                          keras.layers.Conv2D(filters=128,
                                              kernel_size=(3,3),
                                              strides=(1,1),
                                              activation='relu',
                                              kernel_initializer='he_uniform',
                                              data_format='channels_first'),
                                              
                          keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
                          keras.layers.Conv2D(filters=256,
                                              kernel_size=(3,3),
                                              strides=(1,1),
                                              activation='relu',
                                              kernel_initializer='he_uniform',
                                               data_format='channels_first'),
                                              
                          keras.layers.Conv2D(filters=256,
                                              kernel_size=(3,3),
                                              strides=(1,1),
                                              activation='relu',
                                              kernel_initializer='he_uniform',
                                              data_format='channels_first'),
                                              
                          keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
                          keras.layers.Flatten(),
                          keras.layers.Dense(units=1024,
                                             activation='relu',
                                             input_dim=256,
                                             kernel_initializer='he_uniform'),
                          keras.layers.Dense(units=1024,
                                             input_dim=1024,
                                             activation='relu',
                                             kernel_initializer='he_uniform'),
                          keras.layers.Dense(units=10,
                                             input_dim=1024,
                                             activation='softmax',
                                             kernel_initializer='glorot_uniform')
  
  
  ])

  model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(hp.Choice('learningrate',values=[1e-1,1e-2,1e-3])),metrics=['accuracy'])
  model.summary()
  return model



from kerastuner import RandomSearch
import kerastuner.engine.hyperparameters as hp  

tuner_search=RandomSearch(bulid_model,
                          objective='val_accuracy',
                          max_trials=5
                          )

tuner_search.search(X_train,y_train,epochs=3,validation_split=0.1)
model=tuner_search.get_best_models(num_models=1)[0]
model.fit(X_train,y_train,epochs=10,initial_epoch=3,validation_split=0.1)


y_pred=model.predict_classes(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
score=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_pred,y_test)


#loadig my own handwriiten imagr from the drive
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.applications.vgg16 import vgg16
img=load_img('/content/drive/My Drive/Colab Notebooks/DigitRecognitionMnist/IMG_20191231_111452.jpg',target_size=(28, 28))
img = np.expand_dims(img, axis=0)
#img = vgg16.preprocess_input(img)
img = img.reshape(img.shape[1:])
b=[0.3,0.59,0.11]
img=np.dot(img,b)
img=img.reshape(1,1,img.shape[0],img.shape[1])
img=img/255.0

y_got=model.predict_classes(img)



#creating more images for better training
img=load_img('/content/drive/My Drive/Colab Notebooks/DigitRecognitionMnist/IMG_20191231_111452.jpg',target_size=(28, 28))
x=img_to_array(img)
x=x.reshape((1,)+x.shape)
i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='/content/drive/My Drive/Colab Notebooks/DigitRecognitionMnist',save_format='jpeg'):
  i=i+1
  if(i>20):
    break
