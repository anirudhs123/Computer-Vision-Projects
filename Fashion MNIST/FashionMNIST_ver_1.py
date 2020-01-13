from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

pip install -q keras

X_train=pd.read_csv('/content/drive/My Drive/Colab Notebooks/miteshsir/train.csv')
X_val=pd.read_csv('/content/drive/My Drive/Colab Notebooks/miteshsir/val.csv')
X_test=pd.read_csv('/content/drive/My Drive/Colab Notebooks/miteshsir/test.csv')


X_train=X_train.iloc[:,1:]
X_val=X_val.iloc[:,1:]

y_train=X_train.iloc[:,784]
y_val=X_val.iloc[:,784]

X_val=X_val.iloc[:,0:783]
X_train=X_train.iloc[:,0:783]
X_test=X_test.iloc[:,0:783]


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_val=sc.fit_transform(X_val)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import PReLU,ELU,LeakyReLU

classifier=Sequential()
classifier.add(Dense(units=10,activation='relu',input_dim=783,kernel_initializer='he_uniform'))
classifier.add(Dense(units=20,activation='relu',kernel_initializer='he_uniform'))
classifier.add(Dense(units=15,activation='relu',kernel_initializer='he_uniform'))
classifier.add(Dense(units=10,activation='softmax',kernel_initializer='glorot_uniform'))
classifier.compile(optimizer="Adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model_history=classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


plt.plot(model_history.history['acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'],loc='upperleft')
plt.show()

#plots variation in loss function value
plt.plot(model_history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upperleft')
plt.show()

y_pred=classifier.predict_classes(X_val,batch_size=10)



from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_val)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_val)


y_test=classifier.predict_classes(X_test,batch_size=10)
