!pip install keras-tuner
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras

X_test=pd.read_csv("/content/drive/My Drive/Colab Notebooks/miteshsir/test.csv")
X_val=pd.read_csv("/content/drive/My Drive/Colab Notebooks/miteshsir/val.csv")
X_train=pd.read_csv('/content/drive/My Drive/Colab Notebooks/miteshsir/train.csv')

y_train=X_train['label']
y_val=X_val['label']


X_test=X_test.drop(['id'],axis=1)
X_train=X_train.drop(['id','label'],axis=1)
X_val=X_val.drop(['id','label'],axis=1)


X_train=X_train.to_numpy(dtype=int)
X_test=X_test.to_numpy(dtype=int)
X_val=X_val.to_numpy(dtype=int)



mtest=X_test.shape[0]
n=X_test.shape[1]
mtrain=X_train.shape[0]
mval=X_val.shape[0]
import math
n_test=math.sqrt(n)


'''from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_val=sc.fit_transform(X_val)'''

X_train=X_train/255
X_test=X_test/255
X_val=X_val/255

X_train=X_train.reshape((int(mtrain),1,int(n_test),int(n_test)))
X_test=X_test.reshape((int(mtest),1,int(n_test),int(n_test)))
X_val=X_val.reshape((int(mval),1,int(n_test),int(n_test)))
#X_train=np.vstack((X_train,X_val))


from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D
from keras.layers import PReLU,LeakyReLU,ELU
from keras import backend as K

K.common.image_dim_ordering()
K.common.set_image_dim_ordering(dim_ordering='th')

def build_model(hp):
  model=keras.Sequential([
                          keras.layers.Conv2D(filters=64,
                                              kernel_initializer='he_uniform',
                                              strides=(1,1),
                                              activation='relu',
                                              input_shape=(1,int(n_test),int(n_test)),
                                              kernel_size=(3,3),
                                              data_format='channels_first'
                                              ),
                          keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
                          keras.layers.Conv2D(kernel_size=(3,3),
                                              kernel_initializer='he_uniform',
                                              strides=(1,1),
                                              activation='relu',
                                              data_format='channels_first',
                                              filters=128
                                              ),
                          keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
                          keras.layers.Conv2D(filters=256,
                                              kernel_size=(3,3),
                                              kernel_initializer='he_uniform',
                                              strides=(1,1),
                                              activation='relu',
                                              data_format='channels_first'
                                              ),
                          keras.layers.Conv2D(filters=256,
                                              kernel_size=(3,3),
                                              kernel_initializer='he_uniform',
                                              strides=(1,1),
                                              activation='relu',
                                              data_format='channels_first'
                                              ),
                          keras.layers.MaxPool2D(pool_size=(2,2),padding='same'),
                          keras.layers.Flatten(),
                          keras.layers.Dense(units=1024,input_dim=256,activation='relu',kernel_initializer='he_uniform'),
                          keras.layers.Dense(units=1024,input_dim=1024,activation='relu',kernel_initializer='he_uniform'),
                          

                          keras.layers.Dense(units=10,input_dim=1024,activation='softmax',kernel_initializer='glorot_uniform')

                                                  
  ])

  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learningrate',values=[1e-2,1e-3,1e-1,1e-4])),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  model.summary()
  return model

from kerastuner import RandomSearch
import kerastuner.engine.hyperparameters as Hyperparametrs


tuner_search=RandomSearch(build_model,
                          objective='accuracy',
                          max_trials=5,
                          directory="save_dir"
                          )

tuner_search.search(X_train,y_train,epochs=3)

model=tuner_search.get_best_models(num_models=1)[0]
model.fit(X_train,y_train,epochs=10,initial_epoch=3,batch_size=10)

y_pred1=model.predict_classes(X_val)
y_test=model.predict_classes(X_test)
from sklearn.metrics import accuracy_score
score1=accuracy_score(y_pred1,y_val)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred1,y_val)


