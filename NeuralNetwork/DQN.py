import tensorflow as tf
import numpy as np

from NeuralNetwork.CNN import CNN
from NeuralNetwork.ReplayBuffer import RelplayBuffer

class DQN:
    def __init__(self, 
                actionsNum,
                targetUpdateFrequency = 200,
                initEpsilon = 1,
                endEpsilon = 0.05,
                epsilonDecay = 0.9995,
                gamma = 0.95,
                lr = 0.01,
                batchSize = 8
        ):
        # Q network
        self.Q = CNN(actionsNum=actionsNum, inputShape=(128, 128, 1))
        self.Q_target = CNN(actionsNum=actionsNum, inputShape=(128, 128, 1))
        self.Q_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.actionsNum = actionsNum
        # 衰減因子
        self.gamma = gamma
        
        self.targetUpdateFrequency = targetUpdateFrequency
        
        # epsilon
        self.eps = initEpsilon
        self.endEpsilon = endEpsilon
        self.epsilonDecay = epsilonDecay
        
        self.iterations = 0
        
        self.replayBuffer = RelplayBuffer((128, 128, 1), batchSize=batchSize, bufferSize=10000)
    def selectAction(self, state: np.ndarray):
        currEps = self.eps
        self.eps = max(self.eps * self.epsilonDecay, self.endEpsilon)
        if (np.random.uniform(0, 1) > currEps):
            self.Q.trainable = False
            return np.argmax(self.Q(state, training = False))
        else:
            return np.random.randint(self.actionsNum)
    
    def train(self):
        self.Q.trainable = True
        
        state, action, nextState, reward, done = self.replayBuffer.sample()
        
        # with tf.stop_gradient() as stopGrad:
        targetQ = reward + (1 - done) * self.gamma * tf.reduce_max(self.Q_target(nextState, training = False), axis=1, keepdims=True)
        with tf.GradientTape() as tape:
            currentQ = tf.gather(self.Q(state, training=True), np.asarray(action, dtype=np.int32), batch_dims=1)
            # 計算loss
            Q_loss = tf.losses.huber(targetQ, currentQ)
        gradientsOfQ = tape.gradient(Q_loss, self.Q.trainable_variables)
        self.Q_optimizer.apply_gradients(zip(gradientsOfQ, self.Q.trainable_variables))
        
        self.iterations += 1
        if (self.iterations % self.targetUpdateFrequency == 0):
            self.Q_target.set_weights(self.Q.get_weights())