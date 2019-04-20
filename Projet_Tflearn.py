import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from tflearn.layers.normalization import local_response_normalization
%matplotlib inline
TRAIN_DIR = 'H:/data/train_data.npy'
TEST_DIR = 'H:/data/test_data.npy'
IMG_SIZE = 50
# LR = 1e-3 ===> 0.001
MODEL_NAME = 'dogs-vs-cats-convnet'


def create_label(image_name):
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])


train_data = np.load(TRAIN_DIR)
test_data = np.load(TEST_DIR)
train = train_data[:-500]
test = train_data[-500:]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]


tf.reset_default_graph()
network = input_data(shape=[None,IMG_SIZE, IMG_SIZE, 1],name='input')

network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)


# layer 2
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)


# layer 3
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

# layer 4
network = conv_2d(network, 384, 3, activation='relu')
# layer 5
network = conv_2d(network, 384, 3, activation='relu')
# layer 6
network = conv_2d(network, 384, 3, activation='relu')

# layer 7
# network = fully_connected(network, 4096, activation='tanh')
# network = dropout(network, 0.5)

# # layer 8
# network = fully_connected(network, 4096, activation='tanh')
# network = dropout(network, 0.5)

# # layer 9
# network = fully_connected(network, 4096, activation='tanh')
# network = dropout(network, 0.5)

# layer 10
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='momentum', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(network, tensorboard_dir='H:/data/tmp/tflearn_logs', tensorboard_verbose=3)

# model.fit({'input': X_train}, {'targets': y_train}, n_epoch=1, 
#           validation_set=({'input': X_test}, {'targets': y_test}), 
#           snapshot_step=500, show_metric=True, run_id='Alexnet')

# model.save('H:/data/Alexnet_4.tfl')
model.load('H:/data/Alexnet_4.tfl')
pre = model.predict(X_test)
lst_pre = np.array([(pre[i][1]) for i in range(pre.shape[0])])
lst_y_test = np.array([(y_test[i][1]) for i in range(len(y_test))])

# y_test ===> lst_y_test ==> y
# x_test ===> pre ==> X




## Ada_Boost
class FlaCut(object):
    def __init__(self):
        self.mode = 'Undetermined'
        self.th = None
        
        
    def predict(self, data):
        if self.mode == 'horizontal_gt':
            pred = data[:,0] >= self.th
        elif self.mode == 'horizontal_lt':
            pred = data[:,0] < self.th
        elif self.mode == 'vertical_gt':
            pred = data[:,1] >= self.th
        elif self.mode == 'vertical_lt':
            pred = data[:,1] < self.th
        else:
            assert False, "Unknown mode "
            
        return pred.astype(np.float)
    
    def fit(self, data, targets):
        xmin, xmax = data[:,0].min(),data[:,0].max()
        ymin, ymax = data[:,1].min(),data[:,1].max()
        best_th = None
        best_mode = None
        best_accuracy = 0   
        
        for self.mode in ['horizontal_gt','horizontal_lt']:
            for self.th in np.linspace(xmin,xmax,100):
                accu = np.count_nonzero(self.predict(data) == targets)/ float(targets.size)
                if accu > best_accuracy:
                    best_mode = self.mode
                    best_th = self.th
                    best_accuracy = accu
                    
        for self.mode in ['vertical_gt','vertical_lt']:
            for self.th in np.linspace(ymin,ymax,100):
                accu = np.count_nonzero(self.predict(data) == targets)/ float(targets.size)
                if accu > best_accuracy:
                    best_mode = self.mode
                    best_th = self.th
                    best_accuracy = accu
            
        self.th = best_th
        self.mode = best_mode
        print(self.mode, self.th)


def ensemble_pred(X,ensemble):
    B = X.shape[0]
    A0 = np.zeros(N)
    A1 = np.zeros(N)
    
    for p, a in ensemble:
        pred = p.predict(X)
        A0[pred == 0] += a
        A1[pred == 0] += a
        
    return (A1>A0).astype(float)


# y_test ===> lst_y_test ==> y
# x_test ===> pre ==> X

rng = np.random.RandomState(0)
N = pre.shape[0]
subsample_size = N
T = 50
D = np.ones(N)/pre.shape[0]
ind = rng.choice(pre.shape[0], size=(subsample_size,),p=D)
X_= pre[ind]
y_= lst_y_test[ind]
# X_.shape
# y_.shape
ensemble = []
for t in range(T):
    weak_pred_t = FlaCut()
    weak_pred_t.fit(X_, y_)
    
    pred = weak_pred_t.predict(pre)
    errors = float((pred != y)) * D
    error_w = errors * D
    eps_t = max(error_w.sum(), 1e-6)

    alpha = 0.5 * np.log((1-eps_t)/eps_t)
    D *= np.exp((errors - 0.5)* 2.0 * alpha)
    D /= D.sum()
    
    ensemble.append((weak_pred_t, alpha))
    

# layer 7
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)

# layer 8
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)

# layer 9
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)



network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='momentum', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(network, tensorboard_dir='H:/data/tmp/tflearn_logs', tensorboard_verbose=3)

model.fit({'input': X_train}, {'targets': y_train}, n_epoch=1, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
          snapshot_step=500, show_metric=True, run_id='Alexnet')
























































    # #visualise
    # plt.figure(1,(12, 4))
    # plt.subplot(1,3,1)
    # plt.scatter(X_[:,0],X_[:,1], c=weak_pred_t.predict(X_),cmap='summer', s=64)
    # plt.subplot(1,3,2)
    # plt.scatter(pre[:,0], pre[:,1], c=ensemble_pred(pre, ensemble), cmap='summer', s=64)
    # plt.subplot(1,3,3)
    # plt.scatter(pre[:,0],pre[:,1], c=D, cmap='hot' , s=64, vmin=0, vmax=D.max())
    # plt.show()
    
    # ind = rng.choice(N, size=(subsample_size,),p=D)
    # X_= pre[ind]
    # y_= lst_y_test[ind]




# # Load libraries
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn import datasets
# # Import train_test_split function
# from sklearn.model_selection import train_test_split
# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# # # Load data
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # y_test ===> lst_y_test ==> y
# # x_test ===> pre ==> X

# type(X)
# X.shape
# y.shape

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# # Create adaboost classifer object
# abc = AdaBoostClassifier(n_estimators=50,
#                          learning_rate=1)
# # Train Adaboost Classifer
# model = abc.fit(pre, lst_y_test)

# #Predict the response for test dataset
# y_pred = model.predict(pre)

# print("Accuracy:",metrics.accuracy_score(lst_y_test, y_pred))







# model.get_weights(network.W)
# model.summaries.get_summary()
# model.predict(X_train)