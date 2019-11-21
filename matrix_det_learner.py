''' Training a NN how to learn the determinant function of a matrix '''

import numpy as np


### Make Dataset of Random Square Matrices ###

matrix_size = 2
train_size = 10000

# puts determinants into 2 classes [for now] 
def sign_func(A):
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

### Build a Neural Network for Classification

#import keras

from sklearn.neural_network import MLPClassifier

hidden_layer_sizes = (300,2)
''' 98% accuracy with (300,2) and learning rate 0.01'''
nnclf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation='tanh',learning_rate='invscaling',learning_rate_init=0.01,max_iter=1000)
nnclf.fit(X_train,y_train)

#test accuracy
y_pred = nnclf.predict(X_test)

accuracy = sum(y_test == y_pred)/test_size
print('Accuracy: {}'.format(accuracy))

### Build a Neural Network for Regression
'''later'''

