import tensorflow as tf

from NeuralNetwork.CNN import CNN

class DQN:
    def __init__(self, 
                actionsNum,
                discount,
                targetUpdateFrequency = 200,
                initEpsilon = 1,
                endEpsilon = 0.05,
                epsilonDecay = 0.9995,
                gamma = 0.95
        ):
        # Q network
        self.Q = CNN(actionsNum=actionsNum)
        self.targetQ = tf.keras.models.clone_model(self.Q)
        self.Q_optimizer = tf.keras.optimizers.Adam()
        # 衰減因子
        self.discount = discount
        
        self.targetUpdateFrequency = targetUpdateFrequency
        
        # epsilon
        self.initEpsilon = initEpsilon
        self.endEpsilon = endEpsilon
        self.epsilonDecay = epsilonDecay