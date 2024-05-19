import tensorflow as tf
import numpy as np

from NeuralNetwork.CNN import CNN
from NeuralNetwork.ReplayBuffer import RelplayBuffer

class DDQN:
    def __init__(self, 
                actionsNum,
                targetUpdateFrequency = 1000,
                initEpsilon = 1,
                endEpsilon = 0.05,
                epsilonDecay = 0.995,
                gamma = 0.95,
                lr = 0.01,
                batchSize = 8
        ):
        lrSchedule = tf.optimizers.schedules.ExponentialDecay(
            lr, decay_steps=5000, decay_rate=0.96, staircase=True
        )
        # Q network
        self.Q = CNN(actionsNum=actionsNum, inputShape=(128, 128, 1))
        self.Q_target = CNN(actionsNum=actionsNum, inputShape=(128, 128, 1))
        self.Q_optimizer = tf.keras.optimizers.SGD(learning_rate=lrSchedule, clipvalue=0.5)
        self.actionsNum = actionsNum
        # 衰減因子
        self.gamma = gamma
        
        self.targetUpdateFrequency = targetUpdateFrequency
        
        # epsilon
        self.eps = initEpsilon
        self.endEpsilon = endEpsilon
        self.epsilonDecay = epsilonDecay
        
        self.iterations = 0
        self.batchSize = batchSize
        
        self.replayBuffer = RelplayBuffer((128, 128, 1), batchSize=batchSize, bufferSize=10000)
    
    def setEps(self, newEps):
        self.eps = newEps
    
    def loadWeight(self, path: str):
        self.Q.load_weights(path).expect_partial()
        self.Q_target.set_weights(self.Q.get_weights())
    
    def updateEps(self):
        self.eps = max(self.eps * self.epsilonDecay, self.endEpsilon)
        
    def selectAction(self, state: np.ndarray):
        if (np.random.uniform(0, 1) > self.eps):
            self.Q.trainable = False
            return np.argmax(self.Q(state, training = False))
        else:
            return np.random.randint(self.actionsNum)
    
    def train(self):
        self.Q.trainable = True
        
        state, action, nextState, reward, done = self.replayBuffer.sample()
        with tf.GradientTape() as tape:
            Q_eval = self.Q(state, training = True)
            Q_target = Q_eval.numpy()
            Q_eval_next = self.Q(nextState, training = False)
            Q_eval_next = Q_eval_next.numpy()
            argMaxActions = np.argmax(Q_eval_next, axis=1)
            Q_next = self.Q_target(nextState, training = False)
            Q_next = Q_next.numpy()
            index = np.arange(self.batchSize, dtype=np.int32)
            
            # 進行DDQN公式
            Q_target[index, action] = reward + (1 - done) * self.gamma * Q_next[index, argMaxActions]
            Q_loss = tf.keras.losses.huber(Q_target, Q_eval)
        gradientsOfQ = tape.gradient(Q_loss, self.Q.trainable_variables)
        self.Q_optimizer.apply_gradients(zip(gradientsOfQ, self.Q.trainable_variables))
        
        self.iterations += 1
        if (self.iterations % self.targetUpdateFrequency == 0):
            self.Q_target.set_weights(self.Q.get_weights())
        
        return np.mean(Q_loss)