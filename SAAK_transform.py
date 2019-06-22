import tensorflow as tf
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from matplotlib import pyplot as plt 
import keras
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import sys
import pickle
np.random.seed(123)
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
tf.set_random_seed(123)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))
import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()

(feat_train,l_train),(feat_test,l_test) = data
feat_train = feat_train.astype('float32')
feat_test = feat_test.astype('float32')
feat_train/=255
feat_test/=255
pca_train = feat_train[:,:,:]
pca_train=pca_train.reshape(pca_train.shape[0],28,28,1)
print(pca_train.shape)
npad = ((0, 0), (2, 2), (2, 2))
feat_train = np.pad(feat_train, ((0,0),(2, 2), (2, 2)), mode='constant')
feat_test = np.pad(feat_test, ((0,0),(2, 2), (2, 2)), mode='constant') 
print(feat_train.shape)
feat_train = feat_train.astype('float32')
feat_test = feat_test.astype('float32')
x_train = np.empty((60000,0))
x_test = np.empty((10000,0))



#STAGE 1
c=0
#Training
training_data = np.empty((0,4))
for i in range(0,5000):
    j=0
    while(j<=30):
        k=0
        while(k<=30):
            X = np.array([feat_train[i,j,k],feat_train[i,j,k+1],feat_train[i,j+1,k],feat_train[i,j+1,k+1]])
            if(np.std(X)>0):
                training_data = np.append(training_data,[X],axis=0)
                c=c+1
            k=k+2
        j = j+2

print(c)
print(training_data.shape)
pca = PCA(n_components = 3)
pca.fit(training_data)
pca_transf = pca.components_
print(pca_transf.shape)

X_in = feat_train[:,:,:]
X_out = np.zeros((60000,16,16,7))
X_out_test = np.zeros((10000,16,16,7))


#Applying Transform kernel on training and test
for i in range(0,60000):
    j=0
    while(j<=30):
        k=0
        while(k<=30):
            X = np.array([feat_train[i,j,k],feat_train[i,j,k+1],feat_train[i,j+1,k],feat_train[i,j+1,k+1]])
            Y = np.matmul(pca_transf,X)
            Yaug = np.append(Y,np.negative(Y))
            dc = np.sum(X)/2
            Yrelu = np.maximum(Yaug,0)
            j1 = j//2
            k1=k//2
            X_out[i,j1,k1,:] = np.append(Yrelu,dc)
            k=k+2
        j=j+2

for i in range(0,10000):
    j=0
    while(j<=30):
        k=0
        while(k<=30):
            X_test = np.array([feat_test[i,j,k],feat_test[i,j,k+1],feat_test[i,j+1,k],feat_test[i,j+1,k+1]])
            Y_test = np.matmul(pca_transf,X_test)
            Yaug_test = np.append(Y_test,np.negative(Y_test))
            dc_test = np.sum(X_test)/2
            Yrelu_test = np.maximum(Yaug_test,0)
            j1 = j//2
            k1=k//2
            X_out_test[i,j1,k1,:] = np.append(Yrelu_test,dc_test)
            k=k+2
        j=j+2


            
print(X_out.shape)
print(X_out_test.shape)

X_out_flatten_test = X_out_test.reshape(10000,16*16*7)
x_test = np.hstack((x_test,X_out_flatten_test))

X_out_flatten = X_out.reshape(60000,16*16*7)
x_train = np.hstack((x_train,X_out_flatten))            
                
#STAGE 2


X_in = X_out[:,:,:,:]
X_in_test = X_out_test[:,:,:,:]
X_out = np.empty((60000,8,8,9))
X_out_test = np.empty((10000,8,8,9))
c=0           
training_data = np.empty((0,4*7))
for i in range(0,10000):
    j=0
    while(j<=14):
        k=0
        while(k<=14):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            if(np.std(X)>0):
                training_data = np.append(training_data,[X],axis=0)
                c=c+1
            k=k+2
        j= j+2

print(training_data.shape)
pca = PCA(n_components = 4)
pca.fit(training_data)
pca_transf = pca.components_
print(pca_transf.shape)
print(c)

for i in range(0,60000):
    j=0
    while(j<=14):
        k=0
        while(k<=14):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            Y = np.matmul(pca_transf,X)
            Yaug = np.append(Y,np.negative(Y))
            dc = np.sum(X)/2
            Yrelu = np.maximum(Yaug,0)
            j1 = j//2
            k1=k//2
            X_out[i,j1,k1,:] = np.append(Yrelu,dc)            
            k=k+2
        j=j+2

for i in range(0,10000):
    j=0
    while(j<=14):
        k=0
        while(k<=14):
            X_test = np.array([X_in_test[i,j,k,:],X_in_test[i,j,k+1,:],X_in_test[i,j+1,k,:],X_in_test[i,j+1,k+1,:]])
            X_test = X_test.flatten()
            Y_test = np.matmul(pca_transf,X_test)
            Yaug_test = np.append(Y_test,np.negative(Y_test))
            dc_test = np.sum(X_test)/2
            Yrelu_test = np.maximum(Yaug_test,0)
            j1 = j//2
            k1=k//2
            X_out_test[i,j1,k1,:] = np.append(Yrelu_test,dc_test)            
            k=k+2
        j=j+2

print(X_out.shape)
print(X_out_test.shape)

X_out_flatten_test = X_out_test.reshape(10000,8*8*9)
x_test = np.hstack((x_test,X_out_flatten_test))


X_out_flatten = X_out.reshape(60000,8*8*9)
x_train = np.hstack((x_train,X_out_flatten))  

#STAGE 3


X_in = X_out[:,:,:,:]
X_in_test = X_out_test[:,:,:,:]
X_out = np.empty((60000,4,4,15))
X_out_test = np.empty((10000,4,4,15))
c=0

training_data = np.empty((0,4*9))
for i in range(0,20000):
    j=0
    while(j<=6):
        k=0
        while(k<=6):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            if(np.std(X)>0):
                training_data = np.append(training_data,[X],axis=0)
                c=c+1
            k=k+2
        j= j+2

print(training_data.shape)
pca = PCA(n_components = 7)
pca.fit(training_data)
pca_transf = pca.components_
print(pca_transf.shape)
print(c)

for i in range(0,60000):
    j=0
    while(j<=6):
        k=0
        while(k<=6):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            Y = np.matmul(pca_transf,X)
            Yaug = np.append(Y,np.negative(Y))
            dc = np.sum(X)/2
            Yrelu = np.maximum(Yaug,0)
            j1 = j//2
            k1=k//2
            X_out[i,j1,k1,:] = np.append(Yrelu,dc)            
            k=k+2
        j=j+2

for i in range(0,10000):
    j=0
    while(j<=6):
        k=0
        while(k<=6):
            X_test = np.array([X_in_test[i,j,k,:],X_in_test[i,j,k+1,:],X_in_test[i,j+1,k,:],X_in_test[i,j+1,k+1,:]])
            X_test = X_test.flatten()
            Y_test = np.matmul(pca_transf,X_test)
            Yaug_test = np.append(Y_test,np.negative(Y_test))
            dc_test = np.sum(X_test)/2
            Yrelu_test = np.maximum(Yaug_test,0)
            j1 = j//2
            k1=k//2
            X_out_test[i,j1,k1,:] = np.append(Yrelu_test,dc_test)            
            k=k+2
        j=j+2
        
print(X_out.shape)
print(X_out_test.shape)

X_out_flatten_test = X_out_test.reshape(10000,4*4*15)
x_test = np.hstack((x_test,X_out_flatten_test))


X_out_flatten = X_out.reshape(60000,4*4*15)
x_train = np.hstack((x_train,X_out_flatten))  

#STAGE 4


X_in = X_out[:,:,:,:]
X_out = np.empty((60000,2,2,13))

X_in_test = X_out_test[:,:,:,:]
X_out_test = np.empty((10000,2,2,13))

c=0           
training_data = np.empty((0,4*15))
for i in range(0,60000):
    j=0
    while(j<=2):
        k=0
        while(k<=2):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            if(np.std(X)>0):
                training_data = np.append(training_data,[X],axis=0)
                c=c+1
            k=k+2
        j= j+2

print(training_data.shape)
pca = PCA(n_components = 6)
pca.fit(training_data)
pca_transf = pca.components_
print(pca_transf.shape)
print(c)

for i in range(0,60000):
    j=0
    while(j<=2):
        k=0
        while(k<=2):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            Y = np.matmul(pca_transf,X)
            Yaug = np.append(Y,np.negative(Y))
            dc = np.sum(X)/2
            Yrelu = np.maximum(Yaug,0)
            j1 = j//2
            k1=k//2
            X_out[i,j1,k1,:] = np.append(Yrelu,dc)            
            k=k+2
        j=j+2

for i in range(0,10000):
    j=0
    while(j<=2):
        k=0
        while(k<=2):
            X_test = np.array([X_in_test[i,j,k,:],X_in_test[i,j,k+1,:],X_in_test[i,j+1,k,:],X_in_test[i,j+1,k+1,:]])
            X_test = X_test.flatten()
            Y_test = np.matmul(pca_transf,X_test)
            Yaug_test = np.append(Y_test,np.negative(Y_test))
            dc_test = np.sum(X_test)/2
            Yrelu_test = np.maximum(Yaug_test,0)
            j1 = j//2
            k1=k//2
            X_out_test[i,j1,k1,:] = np.append(Yrelu_test,dc_test)            
            k=k+2
        j=j+2

print(X_out.shape)
print(X_out_test.shape)

X_out_flatten_test = X_out_test.reshape(10000,2*2*13)
x_test = np.hstack((x_test,X_out_flatten_test))

X_out_flatten = X_out.reshape(60000,2*2*13)
x_train = np.hstack((x_train,X_out_flatten))  

#STAGE 5


X_in = X_out[:,:,:,:]
X_out = np.empty((60000,1,1,17))

X_in_test = X_out_test[:,:,:,:]
X_out_test = np.empty((10000,1,1,17))

c=0           
training_data = np.empty((0,4*13))
for i in range(0,60000):
    j=0
    while(j<=0):
        k=0
        while(k<=0):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            if(np.std(X)>0):
                training_data = np.append(training_data,[X],axis=0)
                c=c+1
            k=k+2
        j= j+2

print(training_data.shape)
pca = PCA(n_components = 8)
pca.fit(training_data)
pca_transf = pca.components_
print(pca_transf.shape)
print(c)

for i in range(0,60000):
    j=0
    while(j<=0):
        k=0
        while(k<=0):
            X = np.array([X_in[i,j,k,:],X_in[i,j,k+1,:],X_in[i,j+1,k,:],X_in[i,j+1,k+1,:]])
            X = X.flatten()
            Y = np.matmul(pca_transf,X)
            Yaug = np.append(Y,np.negative(Y))
            dc = np.sum(X)/2
            Yrelu = np.maximum(Yaug,0)
            j1 = j//2
            k1=k//2
            X_out[i,j1,k1,:] = np.append(Yrelu,dc)            
            k=k+2
        j=j+2

for i in range(0,10000):
    j=0
    while(j<=0):
        k=0
        while(k<=0):
            X_test = np.array([X_in_test[i,j,k,:],X_in_test[i,j,k+1,:],X_in_test[i,j+1,k,:],X_in_test[i,j+1,k+1,:]])
            X_test = X_test.flatten()
            Y_test = np.matmul(pca_transf,X_test)
            Yaug_test = np.append(Y_test,np.negative(Y_test))
            dc_test = np.sum(X_test)/2
            Yrelu_test = np.maximum(Yaug_test,0)
            j1 = j//2
            k1=k//2
            X_out_test[i,j1,k1,:] = np.append(Yrelu_test,dc_test)            
            k=k+2
        j=j+2

print(X_out.shape)
print(X_out_test.shape)

X_out_flatten_test = X_out_test.reshape(10000,1*1*17)
x_test = np.hstack((x_test,X_out_flatten_test))


X_out_flatten = X_out.reshape(60000,1*1*17)
x_train = np.hstack((x_train,X_out_flatten))  

print(x_train.shape)
print(x_test.shape)

bf = SelectKBest(score_func=f_classif,k=1000)
bf.fit(x_train,l_train)
x_train = bf.transform(x_train)
x_test = bf.transform(x_test)
print(x_train.shape)


#PCA REDUCTION TO 128

print()
print()
print("PCA REDUCTION TO 128")
pca = PCA(n_components = 128)
pca.fit(x_train)
x_train1 = pca.transform(x_train)
x_test1 = pca.transform(x_test)
print(x_train1.shape)


#Classification SVM
model = SVC(max_iter =1000)
#estimator = KerasClassifier(build_fn= model, epochs=20, batch_size=5, verbose=0)
model.fit(x_train1,l_train)

label_train_pred = model.predict(x_train1)
acc_train =  accuracy_score(l_train,label_train_pred)
print("Training Accuracy using SVM Classifier: ",acc_train)
print()
            
label_test_pred = model.predict(x_test1)
acc_test =  accuracy_score(l_test,label_test_pred)
print("Testing Accuracy using SVM CLassifier: ",acc_test)


#Classification Random Forest

model = RandomForestClassifier()
#estimator = KerasClassifier(build_fn= model, epochs=20, batch_size=5, verbose=0)
model.fit(x_train1,l_train)

label_train_pred = model.predict(x_train1)
acc_train =  accuracy_score(l_train,label_train_pred)
print("Training Accuracy using Random Forest Classifier: ",acc_train)
print()
            
label_test_pred = model.predict(x_test1)
acc_test =  accuracy_score(l_test,label_test_pred)
print("Testing Accuracy using Random Forest Classifier: ",acc_test)

#PCA REDUCTION TO 64
print()
print()
print("PCA REDUCTION TO 64")
pca = PCA(n_components = 64)
pca.fit(x_train)
x_train2 = pca.transform(x_train)
x_test2 = pca.transform(x_test)
print(x_train2.shape)


#Classification
model = SVC(max_iter =1000)
#estimator = KerasClassifier(build_fn= model, epochs=20, batch_size=5, verbose=0)
model.fit(x_train2,l_train)

label_train_pred = model.predict(x_train2)
acc_train =  accuracy_score(l_train,label_train_pred)
print("Training Accuracy using SVM Classifier: ",acc_train)
print()
            
label_test_pred = model.predict(x_test2)
acc_test =  accuracy_score(l_test,label_test_pred)
print("Testing Accuracy using SVM CLassifier: ",acc_test)

#Classification Random Forest

model = RandomForestClassifier()
#estimator = KerasClassifier(build_fn= model, epochs=20, batch_size=5, verbose=0)
model.fit(x_train2,l_train)

label_train_pred = model.predict(x_train2)
acc_train =  accuracy_score(l_train,label_train_pred)
print("Training Accuracy using Random Forest Classifier: ",acc_train)
print()
            
label_test_pred = model.predict(x_test2)
acc_test =  accuracy_score(l_test,label_test_pred)
print("Testing Accuracy using Random Forest Classifier: ",acc_test)

#PCA REDUCTION TO 32
print()
print()
print("PCA REDUCTION TO 32")
pca = PCA(n_components = 32)
pca.fit(x_train)
x_train3 = pca.transform(x_train)
x_test3 = pca.transform(x_test)
print(x_train3.shape)


#Classification
model = SVC(max_iter =1000)
#estimator = KerasClassifier(build_fn= model, epochs=20, batch_size=5, verbose=0)
model.fit(x_train3,l_train)

label_train_pred = model.predict(x_train3)
acc_train =  accuracy_score(l_train,label_train_pred)
print("Training Accuracy using SVM Classifier: ",acc_train)
print()
            
label_test_pred = model.predict(x_test3)
acc_test =  accuracy_score(l_test,label_test_pred)
print("Testing Accuracy using SVM CLassifier: ",acc_test)

#Classification Random Forest

model = RandomForestClassifier()
#estimator = KerasClassifier(build_fn= model, epochs=20, batch_size=5, verbose=0)
model.fit(x_train3,l_train)

label_train_pred = model.predict(x_train3)
acc_train =  accuracy_score(l_train,label_train_pred)
print("Training Accuracy using Random Forest Classifier: ",acc_train)
print()
            
label_test_pred = model.predict(x_test3)
acc_test =  accuracy_score(l_test,label_test_pred)
print("Testing Accuracy using Random Forest Classifier: ",acc_test)
