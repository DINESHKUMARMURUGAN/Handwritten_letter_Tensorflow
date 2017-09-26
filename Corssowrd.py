
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import  MaxPooling2D,Convolution2D,Conv2D

from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
from keras import optimizers
from keras.utils import np_utils
import numpy as np
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os
import matplotlib.pyplot as plt
import matplotlib
import theano



path1 = '/Users/dmurugan/Downloads/English/Hnd/Img/'
path2 = '/Users/dmurugan/mess/English/Hnd/'



listing = os.listdir(path1)
num_samples = size(listing)
print(num_samples)

img_rows,img_cols = 200,200

for folder in listing:
    files = os.listdir(path1 + folder + '/')
    for file in files:
        im = Image.open(path1 + folder + '/' + file)
        img = im.resize((img_rows,img_cols))
        gray = img.convert('L')
        gray.save(path2 + '/' + file,"png")


imlist = os.listdir(path2)

im1 = array(Image.open(path2+imlist[0]))
m,n = im1.shape[0:2]
imnbr = len(imlist)

immatrix = array([array(Image.open(path2+im2)).flatten() for im2 in imlist],'f')

label = np.ones((3410,),dtype=int)
label[0:55] = 0 #0
label[55:110] = 1 #1
label[110:165] = 2 #2
label[165:220] = 3 #3
label[220:275] = 4 #4
label[275:330] = 5 #5
label[330:385] = 6 #6
label[385:440] = 7 #7
label[440:495] = 8 #8
label[495:550] = 9 #9
label[550:605] = 10 #A
label[605:660] = 11 #B
label[660:715] = 12 #C
label[715:770] = 13 #D
label[770:825] = 14 #E
label[825:880] = 15 #F
label[880:935] = 16 #G
label[935:990] = 17 #H
label[990:1045] = 18 #I
label[1045:1110] = 19 #J
label[1110:1155] = 20 #K
label[1155:1210] = 21 #L
label[1210:1265] = 22 #M
label[1265:1320] = 23 #N
label[1320:1375] = 24 #O
label[1375:1430] = 25 #P
label[1430:1485] = 26 #Q
label[1485:1540] = 27 #R
label[1540:1595] = 28 #S
label[1595:1650] = 29 #T
label[1650:1705] = 30 #U
label[1705:1760] = 31 #V
label[1760:1815] = 32 #W
label[1815:1870] = 33 #X
label[1870:1925] = 34 #Y
label[1925:1980] = 35 #Z
label[1980:2035] = 36 #a
label[2035:2090] = 37 #b
label[2090:2145] = 38 #c
label[2145:2200] = 39 #d
label[2200:2255] = 40 #e
label[2255:2310] = 41 #f
label[2310:2365] = 42 #g
label[2365:2420] = 43 #h
label[2420:2475] = 44 #i
label[2475:2530] = 45 #j
label[2530:2585] = 46 #k
label[2585:2640] = 47 #l
label[2640:2695] = 48 #m
label[2695:2750] = 49 #n
label[2750:2805] = 50 #o
label[2805:2860] = 51 #p
label[2860:2915] = 52 #q
label[2915:2970] = 53 #r
label[2970:3025] = 54 #s
label[3025:3080] = 55 #t
label[3080:3135] = 56 #u
label[3135:3190] = 57 #v
label[3190:3245] = 58 #w
label[3245:3300] = 59 #x
label[3300:3355] = 60 #y
label[3355:] = 61 #z


data,Label = shuffle(immatrix, label, random_state = 2)
train_date = [data,Label]



#### ACTUAL MODELLING CODE STARTS HERE ###

batch_size = 32
nb_classes = 62
nb_epox = 100

img_channels = 1
nb_filters = 32

nb_pools = 1
nb_conv = 1



(X,y) = (train_date[0],train_date[1])




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')




X_train /= 255
X_test /= 255


y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)


model = Sequential()

model.add(Convolution2D(nb_filters,nb_conv,nb_conv, border_mode='valid', input_shape=(1,img_rows,img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pools,nb_pools)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta')



model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epox,verbose=1,validation_data=(X_test,y_test))




score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score)
print('Test accuracy:',score[1])
print(model.predict_classes(X_test[1:10]))
print(y_test[1:10])


































