import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, actionsNum):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4)
        self.conv1_bnor = tf.keras.layers.BatchNormalization()
        self.conv1Relu = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=3)
        self.conv2_bnor = tf.keras.layers.BatchNormalization()
        self.conv2Relu = tf.keras.layers.ReLU()
        
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1)
        self.conv3_bnor = tf.keras.layers.BatchNormalization()
        self.conv3Relu = tf.keras.layers.ReLU()
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.den1 = tf.keras.layers.Dense(256)
        self.den1Relu = tf.keras.layers.ReLU()
        self.den2 = tf.keras.layers.Dense(64)
        self.den2Relu = tf.keras.layers.ReLU()
        self.den3 = tf.keras.layers.Dense(32)
        self.den3Relu = tf.keras.layers.ReLU()
        self.out = tf.keras(actionsNum)
        
    def call(self, Input):
        x = self.conv1(Input)
        x = self.conv1_bnor(x)
        x = self.conv1Relu(x)
        
        x = self.conv2(x)
        x = self.conv2_bnor(x)
        x = self.conv2Relu(x)
        
        x = self.conv3(x)
        x = self.conv3_bnor(x)
        x = self.conv3Relu(x)
        
        x = self.flatten(x)
        
        x = self.den1(x)
        x = self.den1Relu(x)
        
        x = self.den2(x)
        x = self.den2Relu(x)
        
        x = self.den3(x)
        x = self.den3Relu(x)
        
        Out = self.out(x)
        return Out