''' Training a keras NN how to learn the determinant function of a matrix '''

import numpy as np

### Make Dataset of Random Square Matrices ###

matrix_size = 2
train_size = 10000

def sign_func(A):
	# puts determinants into 2 classes [for now] 
	if A<0: return 0
	#elif A==0: return 0
	else: return 1

X_train = np.random.rand(train_size,matrix_size,matrix_size)
X_train_dets = np.linalg.det(X_train)
clfvec = np.vectorize(sign_func)
y_train = clfvec(X_train_dets)  

#test sampple
test_size = train_size

X_test = np.random.rand(test_size,matrix_size,matrix_size)
X_test_dets = np.linalg.det(X_test)
y_test = clfvec(X_test_dets)  

#reshape matrices to vectors
X_train = X_train.reshape(train_size,matrix_size**2)
X_test = X_test.reshape(test_size,matrix_size**2)

#apply one-hot to labels
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

### Build Model ###

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import elu

hlayer_1 = Dense(250, 
	input_shape=(4,),
	kernel_initializer='random_uniform',
	activation='relu')

hlayer_2 = Dense(100, 
		kernel_initializer='random_uniform',
		activation='relu')

hlayer_3 = Dense(100, 
		kernel_initializer='random_uniform',
		activation='relu')

hlayer_4 = Dense(55, 
		kernel_initializer='random_uniform',	
		activation='relu')

outlayer = Dense(2, 
		kernel_initializer='random_uniform',
		activation='softmax')

model = Sequential([
		hlayer_1,	
		hlayer_2,
		hlayer_3,
		hlayer_4,		
		outlayer
])

model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

### Fit and Test ####

model.fit(X_train,y_train,epochs=50)
