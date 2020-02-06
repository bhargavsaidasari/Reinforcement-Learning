import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
tf.get_logger().setLevel('INFO')


# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

import numpy as np

env = gym.make('CartPole-v0')
obs = env.reset()


class actionProbability(tf.keras.Model):
    #For discrete action space
    def call(self,logits):
        return(tf.squeeze(tf.random.categorical(logits,1),axis=-1))

class policyNetwork(tf.keras.Model):

    def __init__(self,numActions):

        super(policyNetwork,self).__init__()
        #configure size to be a 
        self.hidden1 = layers.Dense(100,activation='relu')  
        #unnormalized log probabilitites bcoz 
        # the categorical distribution accepts those as input
        self.classfier = layers.Dense(numActions)
        self.sample_action = actionProbability()
    
    def call(self,inputs):
        x= self.hidden1(inputs)
        return(self.classfier(x))

    def actionSampling(self,state):
        logits = self.predict_on_batch(state)
        action = self.sample_action(logits)
        return(np.squeeze(action,axis=-1))

policyNet = policyNetwork(env.action_space.n)
env.reset()
action = policyNet.actionSampling(obs[None,:])
print('Action is :' , action )



#class policyGradient():

    #def __init__():

    #def sampleActions(self,actionLogits):
    #    return(tf.squeeze(tf.random_categorical(actionLogits,1),axis=-1))

    #def lossFunction():

    #def buildNetwork():

    #def trainNetwork():