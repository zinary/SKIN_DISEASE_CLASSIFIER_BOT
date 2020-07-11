import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/zinary/Documents/project/images'
TEST_DIR = '/home/zinary/Documents/project/images'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'skin-{}-{}-'.format(LR,'6conv-basic-video')

def label_img(img):
    word_label = img.split(" ")[0]
    if word_label == "Ringworm" : return [1,0]
    elif word_label == "Psoriasis" : return [0,1]
    
def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)) :
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('traindata.npy',training_data)
    return training_data

def process_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split(" ")[0]
        print(img_num)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    print(testing_data)
    shuffle(testing_data)
    np.save('testdata.npy',testing_data)
    return testing_data


       
train_data = create_training_data()
# If you have already created the dataset:
# train_data = np.load('traindata.npy',allow_pickle=True)
print("success")


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
print("success 2")


model.save(MODEL_NAME)


import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_testing_data()
# if you already have some saved:
# test_data = np.load('testdata.npy',allow_pickle=True)

fig=plt.figure()

for num,data in enumerate(test_data[:1]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(1,1,num+1)
    orig = img_data
    
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    print(model_out)
    print(img_num)    
    if np.argmax(model_out) == 1: str_label='Ringworm'
    else: str_label='Psoriasis'
    
    y.imshow(orig,cmap='gray')
    plt.title("found "+str_label+" in "+img_num)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.show()
