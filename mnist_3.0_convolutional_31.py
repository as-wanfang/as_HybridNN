# two-dimensional indicator
# By Fang Wan

import tensorflow as tf
import math
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)
import numpy as np
import pandas as pd

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data 1 - image, 1-deep)       X1 [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]

#                         (input data 2 - indicators, vector)   X2  [batch, 2] two-dimensional indicator
#                                                               W21 [2, 12]
#                                                               Y21 [batch, 7, 7, 12] (tiling)
#                                                               Y31 = Y3 + Y21

#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 11]           B5 [11]
#        · · ·                                                  Y [batch, 11]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
X2 = tf.placeholder(tf.float32, [None, 2])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 11])
# variable learning rate
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W21 = tf.Variable(tf.truncated_normal([2, 12], stddev=0.1))
B21 = tf.Variable(tf.ones([12])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 11], stddev=0.1))
B5 = tf.Variable(tf.ones([11])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

Y21 = tf.nn.relu(tf.matmul(X2, W21) + B21)
Y21 = tf.reshape(Y21,[-1, 1, 1, 12])
Y21 = tf.tile(Y21, [1, 7, 7, 1])
Y3 = Y3 + Y21

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

# Setting up the indicators
checkpoint_path = './checkpoint_1-3'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
np.random.shuffle(id)
n = 2 #n=2, 5, 10
#indicators = {i:id[round(10/n*i):round(10/n*(i+1))] for i in range(n)} #4:n=2 #6:n=5 #7:n=10
#indicators = {0:id[0:3], 1:id[3:10]}        #3
indicators_0 = {0:[0,1,2,3,4], 1:[5,6,7,8,9]} #1
indicators_1 = {0:[0,2,4,6,8], 1:[1,3,5,7,9]} #2
#indicators = {0:[0], 1:[1,2,3,4,5,6,7,8,9]} #5
l1 = len(indicators_0)
l2 = len(indicators_1)
n = l1 * l2

np.savez(checkpoint_path+'/indicators', indicators_0 = indicators_0, indicators_1 = indicators_1)

# traverse all indicators and generate new labels
def generate_data(batch_X, batch_Y):
    l = len(batch_X)
    batch_X_rep = np.tile(batch_X, [n,1,1,1])
    # [0,0],...,[0,0],[1,0],...,[1,0],[0,1],...,[0,1],[1,1],...,[1,1]
    batch_X2_rep = np.tile( np.array([[i,j] for j in range(2) for i in range(2)]).reshape([1, n, 2]), [l,1,1] ).reshape([l*n,2], order='F')

    batch_id = np.argmax( batch_Y, axis=1 ).reshape([l,1])
    batch_id_rep = np.tile(batch_id, [n,1])

    # find the true indicator of each image
    batch_X2_rep_true = np.tile(
        np.array(list(map(lambda x: [[k0,k1] for k0, v0 in indicators_0.items() for k1, v1 in indicators_1.items() if x in v0 and x in v1], batch_id))).squeeze() ,
        [n,1])

    # generate the new labels including the conflict class
    batch_Y_rep = np.zeros((l*n,10+1))
    for i, match in zip(np.arange(l*n),batch_X2_rep_true==batch_X2_rep):
        if match.all():
            batch_Y_rep[i][batch_id_rep[i]] = 1
        else:
            batch_Y_rep[i][10] = 1
    return batch_X_rep, batch_X2_rep, batch_Y_rep

# You can call this function in a loop to train the model, 100 images at a time
a_train = []
a_test = []
test_prediction = []
for i in range(10001):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X_rep, batch_X2_rep, batch_Y_rep = generate_data(batch_X, batch_Y)
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    sess.run(train_step, {X: batch_X_rep, X2: batch_X2_rep, Y_: batch_Y_rep, lr: learning_rate})
    # compute training values for visualisation
    if i%10==0:
        a = sess.run(accuracy, {X: batch_X_rep, X2: batch_X2_rep, Y_: batch_Y_rep})
        print(str(i) + ": accuracy:" + str(a) + " (lr:" + str(learning_rate) + ")")
        a_train.append(a)
    # compute test values for visualisation
    if i%100==0:
        test_X_rep, test_X2_rep, test_Y_rep = generate_data(mnist.test.images, mnist.test.labels)
        a = sess.run(accuracy, {X: test_X_rep, X2: test_X2_rep, Y_: test_Y_rep})
        a_test.append(a)
        y = sess.run(Y, {X: test_X_rep, X2: test_X2_rep})
        test_prediction.append(y)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a))

saver.save(sess, checkpoint_path + '/Network')

print("max test accuracy: " + str(max(a_test)))
np.savez(checkpoint_path+'/accuracy', a_test=a_test, a_train=a_train, test_prediction=test_prediction)

# best 0.994525 after 10000 iterations

# calculate prediction accuracy
y = np.argmax(mnist.test.labels, 1)

dir = 'checkpoint_1-3'
data = np.load(dir+'/accuracy.npz')
indicators = np.load(dir+'/indicators.npz')
indicators_0 = indicators['indicators_0'].item()
indicators_1 = indicators['indicators_1'].item()
l1 = len(indicators_0)
l2 = len(indicators_1)
n = l1 * l2
a_test = data['a_test'] #[101,1]
y_ = data['test_prediction'][np.argmax(a_test)] #[10000*n,11]

# 0 = [0,0], 1=[1,0], 2=[0,1], 3=[1,1]
indicators = {0:[0,2,4], 1:[6,8], 2:[1,3], 3:[5,7,9]}
cols = [ "Indicator_"+str(i) for i in range(n)]
y_prediction = pd.DataFrame(np.zeros([10000, n]), columns=cols)
for i in range(n):
    y_prediction['Indicator_'+str(i)] = np.argmax( y_[i*10000:(i+1)*10000,:],1)

true_count = 0.0
pass_logic_check_count = 0.0
for i in range(10000):
    critirier = y_prediction.loc[i] == 10
    if sum(critirier) != n-1:
        continue
    if y_prediction.loc[i,np.argmin(critirier)] not in indicators[int(np.argmin(critirier).split('_')[1])]:
        continue
# 99.48% pass the logic check in case #1
    pass_logic_check_count += 1
    if y_prediction.loc[i,np.argmin(critirier)] == y[i]:
        true_count += 1

print('The accuracy without logic is %s'%(max(a_test)))
print('The rate of passing logic check is %s'%(pass_logic_check_count/10000))
print('The accuracy with logic check is %s'%(true_count/10000))
print('The accuracy after passing logic check is %s'%(true_count/pass_logic_check_count))
