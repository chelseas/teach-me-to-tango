from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import *
from keras import regularizers
import numpy as np
from scipy import misc
import os

# data
H, W, C = 100, 75, 1

tiny_count = 200 # for testing, don't load all images
count=0
n_train_total = 0
# read in images from file
cwd = os.getcwd()
print("Loading train images...")
Xtrain = np.zeros((0,H,W,C))
for filename in os.listdir("data/tango_images_part1_small"):
	if filename.endswith(".png"):
		n_train_total += 1
		if count <= tiny_count:
			im = misc.imread(os.path.abspath("data/tango_images_part1_small/"+filename), flatten=True)
			im = np.reshape(im, (1,)+im.shape+(1,))
			Xtrain = np.append(Xtrain, im, axis=0)
			count += 1

count=0
print("Loading test images...")
Xtest = np.zeros((0,H,W,C))
for filename in os.listdir("data/tango_images_part2_small"):
	if filename.endswith(".png"):
		if count <= tiny_count:
			im = misc.imread(os.path.abspath("data/tango_images_part2_small/"+filename), flatten=True)
			im = np.reshape(im, (1,)+im.shape+(1,))
			Xtest = np.append(Xtest, im, axis=0)
			count += 1

print("Loading labels...")
# read in labels from file
y_true = np.loadtxt("data/feature_data.csv", delimiter=',', skiprows=1, usecols=(2,3,4))
n_train = len(Xtrain)
n_test = len(Xtest)
y_train = y_true[0:n_train,:]
y_test = y_true[n_train_total:n_train_total+n_test, :]
print(n_train)
print(len(y_train))
print(n_test)
print(len(y_test))
#print(asdfsdfa)

print("Building model...")
# model
model = Sequential()
n_filters = 64
filter_size = 5
model.add(Conv2D(n_filters, 
				filter_size, 
				strides=1,
				padding='same',
				data_format='channels_last', 
				use_bias=True, input_shape=(H,W,C), 
				kernel_regularizer=regularizers.l2(0.01),
				activity_regularizer=regularizers.l1(0.01)))
model.add(Conv2D(128, 
				filter_size, 
				strides=1,
				padding='same',
				data_format='channels_last', 
				use_bias=True,  
				kernel_regularizer=regularizers.l2(0.01),
				activity_regularizer=regularizers.l1(0.01)))
model.add(Conv2D(256, 
				1, 
				strides=1,
				padding='same',
				data_format='channels_last', 
				use_bias=True, 
				kernel_regularizer=regularizers.l2(0.01),
				activity_regularizer=regularizers.l1(0.01)))
model.add(Flatten()) # convert 3D activation maps to 1D feature vectors
model.add(Dense(3,
				kernel_regularizer=regularizers.l2(0.01),
				activity_regularizer=regularizers.l1(0.01))) 
				# FC layer. Produce error rate in x, y, and z

optimizer = Nadam(lr = 2e-3)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

print("Training model...")
batch_size = 32
epochs = 5
model.fit(Xtrain,y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle='batch')

test_loss = model.evaluate(Xtest, y_test, batch_size=batch_size, verbose=1)
print(model.metrics_names[0], ":", test_loss[0])
print(model.metrics_names[1], ":", test_loss[1])


