import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers as tko
from tensorflow.keras import losses as tkl
from matplotlib import pyplot as plt
print(tf.__version__)

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
tf.get_logger().setLevel('INFO')


# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

import numpy as np
import pandas as pd

env = gym.make('CartPole-v0')
obs = env.reset()

#sampling actions from logits
class actionProbability(tf.keras.Model):
    #For discrete action space
    def call(self,logits):
        return(tf.squeeze(tf.random.categorical(logits,1),axis=-1))

# The model
# Input - 100 layer RELU - output-Linear

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
        return(np.squeeze(action,axis=-1),logits)



class pgAgent:

    def __init__(self,model,alpha):
        #model - 
        self.model = model
        self.optimizer = tko.Adam()
    
    def testModel(self,env):

        obs ,isDone , reward = env.reset() ,False, 0

        while(not isDone):
            action = self.model.actionSampling(obs[None,:])
            obs,currentReward,isDone,_ = env.step(action)
            reward = reward + currentReward
            env.render()
        return(reward)

    def _lossFunction(self,actions,logits):
        
        loss = tkl.SparseCategoricalCrossentropy(from_logits=True)
        return(tf.reduce_mean(loss(actions,logits)))
    
    def _trainStep(self,env,episodes = 1):

        #Iterate through episodes accumulating gradients using gradient tape
        #and run the optimizer at the end of all the episodes

        for i in range(episodes):
            epVariables = []
            
            obs ,isDone , epReward = env.reset() ,False, 0
            while(not isDone):
                with tf.GradientTape() as tape:
                    action,logits = self.model.actionSampling(obs[None,:])
                    loss = self._lossFunction(action,logits)
        
                grads = tape.gradient(loss,self.model.trainable_variables)
                obs,currentReward,isDone,_ = env.step(action)
                epReward = epReward + currentReward
                epVariables.append([grads,currentReward])
            
            epVariables = np.array(epVariables)
            epVariables[:,1] = self._valuesComputation(epVariables[:,1])
            
            if(i==0):
                gradBuffer = self.model.trainable_variables
                for counter,grad in enumerate(gradBuffer):
                    gradBuffer[counter] = grad * 0

            for grads,values in epVariables:
                for counter,grad in enumerate(grads) :
                    gradBuffer[counter] += grad * values 
            
        self.optimizer.apply_gradients(zip(gradBuffer, self.model.trainable_variables))
            
            
        return(epReward)
            


    def _valuesComputation(self,rewards,gamma=0.9):
        #Compute Values at each timestep based on the rewards
        valueList = np.zeros_like(rewards)
        previousRewards = 0
        for i in reversed(range(0,len(rewards))):
            valueList[i] = rewards[i]+ previousRewards * gamma
            previousRewards = valueList[i]
        return(valueList)


    def trainLoop(self,env,numEpochs,episodes=1):
        #Run trainstep for the designated amount of epochs

        value= []
        for epochs in range(numEpochs):
            value.append(self._trainStep(env,episodes))
            
            print("Epoch :" + str(epochs) + " Score " + str(value[-1]))
        return(value)



if __name__ == "__main__":
    #Initialize the Neural Network for Policy Gradient
    policyNet = policyNetwork(env.action_space.n)
    #Initialize the agent
    agent = pgAgent(policyNet,alpha=0.2)
    episodeValue = []
    #Train and save weights
    episodeValue = (agent.trainLoop(env,500,10))
    agent.model.save_weights('policyGrad_noadvantage',save_format='tf')

    episodeValue = pd.DataFrame(episodeValue)
    episodeValue.to_csv('valueEpisode.csv')
