#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:54:09 2020

@author: preethasaha
"""

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D , Conv2DTranspose

from keras.models import Sequential, Model
from keras.optimizers import Adam

N_iter=5000
batch_size = 128
check_interval = 10

img_x = 28
img_y = 28
c = 1 

# Input image dimensions
img_shape = (img_x, img_y, c)

# Size of the noise vector, used as input to the Generator
Noise_vec = 100

def sample_images(G, image_grid_rows=4, image_grid_columns=4):

    
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, Noise_vec))

    
    gen_imgs = G.predict(z)

    # Rescaling image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    #image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
def gen(z=(Noise_vec)):
    
    input_l = Input(z)
    hid_l = Dense(7*7*256)(input_l)
    hid_l = Reshape((7, 7, 256))(hid_l)
    
    
    hid_l = Conv2DTranspose(128, kernel_size=[3,3],strides=[2,2],padding="same")(hid_l)
    hid_l = BatchNormalization()(hid_l)
    hid_l = LeakyReLU(alpha=0.01)(hid_l)
    
    hid_l = Conv2DTranspose(64, kernel_size=[3,3],strides=[1,1],padding="same")(hid_l)
    hid_l = BatchNormalization()(hid_l)
    hid_l = LeakyReLU(alpha=0.01)(hid_l)
    
    hid_l = Conv2DTranspose(1, kernel_size=[3,3],strides=[2,2],padding="same")(hid_l)
    output_l = Activation("tanh")(hid_l)
    
    model = Model(inputs=input_l, outputs=output_l)
    model.summary()
  
    return model
    
def disc(input_shape=(img_x, img_y,1)):
    # 64x64x3 -> 32x32x32
    input_l = Input((img_x, img_y,1))
    hid_l = Conv2D(filters=32,kernel_size=[3,3],strides=[2,2],padding="same")(input_l)
    #hid_l = BatchNormalization()(hid_l)
    hid_l = LeakyReLU(alpha=0.01)(hid_l)   
    
    # 32x32x32-> 16x16x64 
    hid_l = Conv2D(filters=64,kernel_size=[3,3],strides=[2,2],padding="same")(input_l)
    hid_l = BatchNormalization()(hid_l)
    hid_l = LeakyReLU(alpha=0.01)(hid_l)   
    
    # 16x16x64  -> 8x8x128  
    hid_l = Conv2D(filters=128,kernel_size=[3,3],strides=[2,2],padding="same")(input_l)
    hid_l = BatchNormalization()(hid_l)
    hid_l = LeakyReLU(alpha=0.01)(hid_l)   
  
    
    hid_l = Flatten()(hid_l)

    output_l = Dense(1, activation='sigmoid')(hid_l)
    model = Model(inputs= input_l, outputs=output_l)

    model.summary()

    return model

def gan_setup(G, D):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(G)
    model.add(D)
    
    model.summary

    return model
    
#Discriminator
D = disc((img_x, img_y,1))
D.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

#only train the generator
D.trainable = False

#Generator
G = gen((Noise_vec,))

#GAN

gan = gan_setup(G,D)


gan.compile(loss='binary_crossentropy',optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoints = []

def train_gan(N_iter, batch_size, check_interval):
    (X_train,_),(_,_)= mnist.load_data()
    print(X_train.shape)
    X_train = X_train / 127.5 - 1.0 
    X_train = np.expand_dims(X_train, axis=3)
    
    label_real = np.ones((batch_size, 1))
    label_fake = np.zeros((batch_size, 1))

    for epoch in range(N_iter):
        
        
        indices  = np.random.randint(0,X_train.shape[0],batch_size)
        all_images = X_train[indices]
        
        z = np.random.normal(0, 1, (batch_size, 100))
        generated_images = G.predict(z)
    
    
        #  real data: discriminator
        d_loss_real = D.train_on_batch(all_images, label_real)


      
        # generated data: discriminator
        d_loss_fake = D.train_on_batch(generated_images,label_fake)
        
        d_loss,accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        
        
        
        # Train generator
        z = np.random.normal(0, 1, (batch_size, 100))
        generated_images = G.predict(z)
        g_loss = gan.train_on_batch(z,label_real)
        
        
        
        if epoch > 0 and (epoch+1) % check_interval == 0:
            
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(epoch + 1)
            
            
            sample_images(G)
        
    
train_gan(N_iter, batch_size, check_interval)    
losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()  



accuracies = np.array(accuracies)

# Plot Discriminator accuracy
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()
    
    
    
    
