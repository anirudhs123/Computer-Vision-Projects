import numpy as np
import keras
import tqdm
import tensorflow as tf
import utils
from tqdm import tqdm

# the DCGAN class containing all the helper functions!!

class DCGAN():
    
    # the init function which is called automatically when we initialize an object of the class
    def __init__(self, learning_rate, batch_size, num_epochs, save_path):
        # the input, output specific parameters!!
        self.img_shape = (28,28,1)
        self.noise_dim = 100
        
        # model specific parameters!!
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_path = save_path # where we save the models weights in a regular basis.
        
        # defining the generator and discriminator. 
        """
        Use the helper functions in this class to define the generator and discriminator!!!
        Use a sequential model to define the gan_model using keras.Sequential to connect the generator to the discriminator.
        
        YOUR CODE GOES HERE .......        
        
        """
        
        # compile the models - I have done this for you .....................
        """
        BONUS : what is binary cross entropy and why is it used here. I am asking very simple stuff :P (easy bonus)
        """
        self.discriminator.compile(keras.optimizers.Adam(10*self.lr), "binary_crossentropy")
        self.gan_model.compile(keras.optimizers.Adam(self.lr), "binary_crossentropy")

    """
    create_gen function to define the generator model.
    Use the keras Sequential models to define the generator. Visit keras documentation page for more info!!
    
    For example:
    model = keras.Sequential()
    layer1 = keras.layers.Dense(n, activation="relu")
    model.add(layer1)
    return model
    """
    def create_gen(self):
        """
        model = keras.Sequential()
        layer1 = ...
        layer2 = .....
        layer3 = .....
        ..
        
        
        model.add(layer1)
        model.add(layer2)
        ....
        
        
        return model
        
        
        YOUR CODE GOES HERE!!!!
        """

        print("Generator model = ")
        print(model.summary())
        return model
    
    """
    create_disc function is used to create the discriminator for the model. 
    Use the same method as above to define the model.
    
    """

    def create_disc(self):
        """
        model = keras.Sequential()
        layer1 = ...
        layer2 = .....
        layer3 = .....
        ..
        
        
        model.add(layer1)
        model.add(layer2)
        ....
        
        
        return model
        
        
        YOUR CODE GOES HERE!!!!
        """
        
        print("Discriminator model = ")
        print(model.summary())
        return model

    # function to pretrain the discriminator model to identify real and fake images!!!
    def pretrain(self, x):
        # how many examples to choose
        size = x.shape[0]//200

        """
        randomly choose size number of images from x - real images
        Similarly create size number of fake images - how will you do this ! (brain time :P)
        
        Now decide on a convention - Lets say - 0 - Fake images, 1 - Real images
        
        For example:
        real_img = ...
        fake_img = ...
        x = np.concatenate(real_img, fake_img)
        labels = [1]*size + [0]*size - x contains first size number of real images (label 1) and next size number of fake images (label 0)
        self.discriminator.fit(x, labels, batch_size = self.batch_size, epochs = 1)
        
        
        
        YOUR CODE GOES HERE!!
        """


        print("Pretraining done!")
        
        
    # function to generate data for the discriminator to train on in every iteration!    
    def gen_data(self, x, size):
        """
        randomly choose size number of real images from x - real images 
        also create size number of fake images - think think !!!!!
        concatenate them and create labels for the same and hence return them
        
        
        return x, labels
        
        YOUR CODE GOES HERE!!!
        
        """
    
    # the training function. Loop through every epoch and perform multiple iterations in each epoch with each iteration for a batch-size number of images!!
    def train(self, x, num_iter):
        # iterating through num_epochs!!
        
        for i in range(self.num_epochs):
            print("Epoch no :"+str(i+1)+"/"+str(self.num_epochs))
        
        
            # iterating through num of iterations!!!
            for j in tqdm(range(num_iter)):
                
                # use the gen_data function to create input and output for the discriminator!!
                x,y = self.gen_data(x, self.batch_size//2)
                # train the discriminator on this data!!!
                self.discriminator.train_on_batch(x,y)
                # Freeze the discriminator to train the GAN model - fix the weights!!
                utils.make_trainable(self.discriminator, False)
                
                # train the gan model!!!!!!!!
                self.gan_model.train_on_batch("""pass the inputs here :P not going to let you go so easily haha!""")
                
                # make the discriminator params back to trainable for the next iteration!!!
                utils.make_trainable(self.discriminator, True)
            
            #save the weights and plot the results every 10 epochs!!!
            if i%10 == 0:
                self.gan_model.save_weights(self.save_path + str(i+1)+".h5")
                utils.plot(self.generator)