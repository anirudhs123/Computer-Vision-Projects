from numpy import expand_dims,zeros,ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Reshape,Flatten,MaxPool2D
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dropout
from tensorflow.keras.layers import ReLU,LeakyReLU,PReLU,ELU
from matplotlib import pyplot
 
# discriminator model
#Here we use channel_last (as 1 is in the last)
def discriminator(in_shape=(28,28,1)):
 model= Sequential()
 model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(2,2),activation='relu',kernel_initializer='he_uniform',input_shape=in_shape))
 #model.add(MaxPool2D(pool_size=(2,2),padding='same'))
 model.add(Dropout(0.2))
 model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(2,2),activation='relu',kernel_initializer='he_uniform'))
 #model.add(MaxPool2D(pool_size=(2,2),padding='same'))
 model.add(Dropout(0.2))
 #model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(2,2),activation='relu',kernel_initializer='he_uniform',input_shape=in_shape))
 #model.add(MaxPool2D(pool_size=(2,2),padding='same'))
 #model.add(Dropout(0.2))
 #model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(2,2),activation='relu',kernel_initializer='he_uniform',input_shape=in_shape))
 #model.add(MaxPool2D(pool_size=(2,2),padding='same'))
 #model.add(Dropout(0.2))
 model.add(Flatten())
 model.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))
 opt = Adam(lr=0.0002,beta_1=0.5)
 model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
 return model   
	

# generator model
def generator(latent_dim):
	model = Sequential()
	#Input for 14x14 image
	n_nodes = 128 * 14 * 14
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((14, 14, 128)))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	#Tried tanh also but as the preprocessed image has pixel values between [0,255] we use sigmoid 
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
 
# combined generator and discriminator model, for updating the generator
def gan(g_model, d_model):
	d_model.trainable = False
	model = Sequential()
	# add generator and add the discriminator
	model.add(g_model)
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002,beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# load and prepare  images
def load_real_images():
	(trainX, _), (_, _) = load_data()
	X = expand_dims(trainX, axis=-1)
	X = X.astype('float32')
	X = X / 255.0
	return X
 
# Generate real samples
def generate_real_images(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_images(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y
 

 # evaluate the discriminator performance
def performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	#get real examples and evaluvate on these examples
	X_r, y_r = generate_real_images(dataset, n_samples)
	_, acc_real = d_model.evaluate(X_r, y_r, verbose=0)
	#get fake examples and evaluvate on fake examples
	#Returns the loss value & metrics values for the model in test mode
	x_f, y_f = generate_fake_images(g_model, latent_dim, n_samples)
	_, acc_fake = d_model.evaluate(x_f, y_f, verbose=0)
	# Printing discriminator performance
	print('Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# Saving the generator model tile file
	filename = 'GAN_%03d.h5' % (epoch)
	g_model.save(filename)
 
# train the generator and discriminator
def train_model(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, batch_size=256):
	batch_per_epoch = int(dataset.shape[0] / batch_size)
	half_batch = int(batch_size / 2)
	
	for i in range(epochs):
		# going through all the batches over the training set
		for j in range(batch_per_epoch):
			# get randomly selected real samples and fake samples
			X_r, y_r = generate_real_images(dataset, half_batch)
			X_f, y_f = generate_fake_images(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_r, X_f)), vstack((y_r, y_f))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, batch_size)
			# create inverted labels for the fake samples
			y_gan = ones((batch_size, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, d_loss, g_loss))
		# evaluate the model performance, sometimes

	#Printing the final real and fake accuracy after 100 epochs
	performance(epochs, g_model, d_model, dataset, latent_dim)
 
# size of the latent space
latent_dim = 100
# Form the discriminator
d_model = discriminator()
# Form the generator
g_model = generator(latent_dim)
# Form the gan
gan_model = gan(g_model, d_model)
# load image data
data = load_real_images()
# train model
train_model(g_model, d_model, gan_model, data, latent_dim)

# create and save a plot of generated images
def img_plot(examples, n):
	# plot images
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray')
	pyplot.show()

from tensorflow.keras.models import load_model 
import numpy as np
# load model
model = load_model('GAN_100.h5')
# generate images
latent_points = generate_latent_points(100, 4)
# generate images
X = model.predict(latent_points)
X=np.around(X)
# plot the result
img_plot(X, 2)

