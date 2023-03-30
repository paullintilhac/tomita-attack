import numpy as np
import sys
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi
x_test = torch.load('/users/paullintilhac/Downloads/x_test_tomita6.pt').numpy()
y_test = torch.load('/users/paullintilhac/Downloads/y_test_tomita6.pt').numpy()
x_train = torch.load('/users/paullintilhac/Downloads/x_train_tomita6.pt').numpy()
y_train = torch.load('/users/paullintilhac/Downloads/y_train_tomita6.pt').numpy()

BUFFER_SIZE=10000
BATCH_SIZE=64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train-0.5, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test-0.5, y_test)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


with tf.Session() as sess:
    model = tf.keras.models.load_model('/users/paullintilhac/downloads/Tomita-6-Keras-perfect.keras',compile=False)
    model.image_size = 6
    model.num_channels = 1
    model.num_labels = 2

    attack = CarliniL2(sess, model, batch_size=1000, max_iterations=1000, confidence=0, targeted=True)

    inputs, targets = generate_data(data, samples=1000, targeted=True, start=0, inception=False)
    #show(inputs)
    timestart = time.time()
    adv = attack.attack(inputs, targets)
    timeend = time.time()

    print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

    for i in range(len(adv)):
        print("Valid:")
        show(inputs[i])
        print("Adversarial:")
        show(adv[i])

        print("Original Classification:", model.model.predict(inputs[i:i + 1]))
        print("Adversarial Classification:", model.model.predict(adv[i:i + 1]))
        distortion = np.sum((adv[i] - inputs[i]) ** 2) ** .5
        print("Total distortion:", distortion)
        dis.append(distortion)

    Original_classification = test_RNN(inputs)
    Adversarial_classification = test_RNN(adv)

    print("Original", Original_classification)
    print("Adv", Adversarial_classification)

    count = 0
    list1 = Original_classification.tolist()
    list2 = Adversarial_classification.tolist()
    for i in range(len(Original_classification)):
        if (list1[i] != list2[i]):
            count = count + 1
    print("Total Number of sample processed", len(Original_classification))
    print("L2 Accuracy :", count/len(Original_classification))
    print("Average Distortion", sum(dis)/len(dis))