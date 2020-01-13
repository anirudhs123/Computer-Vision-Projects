
from PIL import Image
from matplotlib.pyplot import imshow, plot

!pip install keras-tuner


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras


from google.colab import drive
drive.mount('/content/drive')

X=np.load('/content/drive/My Drive/Colab Notebooks/pokemon/poke_image_data.npy')
dataset=pd.read_csv("/content/drive/My Drive/Colab Notebooks/pokemon/names_and_strengths.csv")
y=np.zeros(6036,dtype=int)
i=0
  
for j in range(1,6036):
    if(dataset.name[j]==dataset.name[j-1]):
      y[j]=i
    else:
      i=i+1
      y[j]=i 


b=[0.3,0.59,0.11]
X=np.dot(X,b)
X=X/255      

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D
from keras.layers import PReLU,LeakyReLU,ELU

def build_model(hp):
  model=keras.Sequential([
  keras.layers.Conv2D(
      filters=hp.Int('conv_1_filter',min_value=32,max_value=128,step=16),
      kernel_size=hp.Choice('conv_1_kernel',values=[3,5]),
      activation='relu',
      input_shape=(X.shape[1],X.shape[2],1)
  ),
  keras.layers.Conv2D(
      filters=hp.Int('conv_2_filter',min_value=32,max_value=64,step=16),
      kernel_size=hp.Choice('conv_2_kernel',values=[3,5]),
      activation='relu',
      
  ),
  keras.layers.Flatten(),
  keras.layers.Dense(units=hp.Int('dense_1_units',min_value=32,max_value=128,step=16),activation='relu'),
  keras.layers.Dense(units=806,activation='softmax')
  ])
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-3])),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


  return model


from kerastuner import RandomSearch
import kerastuner.engine.hyperparameters as Hyperparametrs


tuner_search=RandomSearch(build_model,
                          objective='val_acc',
                          max_trials=5,
                          directory='output',
                          project_name='Pokemon')



tuner_search.search(X_train,y_train,epochs=3,validation_split=0.1)

model=tuner_search.get_best_models(num_models=1)[0]

model.fit(X_train,y_train,epochs=15,validation_split=0.1,initial_epoch=3)                          

