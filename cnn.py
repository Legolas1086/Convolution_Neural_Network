import numpy as np 
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

#Training Set Preprocessing
idg_train=ImageDataGenerator(rescale=1./250,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)
training_set=idg_train.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32, class_mode='binary')

#Test_set Preprocesssing

idg_test=ImageDataGenerator(rescale=1./250)
validation_set=idg_test.flow_from_directory('dataset/test_set',target_size=(64,64),class_mode='binary',batch_size=32)

#CNN Build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

network=Sequential()

#First Convolution and maxpool layer
network.add(Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[64,64,3]))
network.add(MaxPool2D(pool_size=2,strides=2))

#Second Convolution and Maxpool Layer
network.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
network.add(MaxPool2D(pool_size=2,strides=2))

#Flatten the output of maxpool layer
network.add(Flatten())
 
#Fully Connected Layer
network.add(Dense(units=128,activation='relu'))
#Add dropout if wanted

#Output layer
network.add(Dense(units=1,activation='sigmoid'))


#Compiling the network
network.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the network
network.fit(x=training_set,validation_data=validation_set,epochs=25)


#Testing against custom input
from keras.preprocessing import image
img=image.load_img('dataset/singleprediction/cat_or_dog_1.jpg',target_size=(64,64))
img=image.img_to_array(img)
img.np.expand_dims(img,axis=0)
y=network.predict(img)
training_set.class_indices

if y[0][0]==1:
    print('dog')
else:
    print('cat')    


