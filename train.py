import tensorflow as tf
import numpy as np
import csv
import datetime

from Agent.Agent import Agent
from Environment.Environment import Environment
from config import *
from NeuralNetwork.DQN import DQN

def enableGPU():
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

def train():
    try:
        enableGPU()
        agent = Agent()
        # # env = Environment(agent, ip="192.168.168.13")
        env = Environment(agent, BatchSize, True, ip="localhost")
        model = DQN(len(actionMap), lr = LearningRate, batchSize=BatchSize)
        
        with open(f"./trainingHistroy/trainHistory_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for episode in range(1, Episodes + 1):
                print(f"Current eps: {model.eps}")
                turnReward, runTime = env.runOneEpisode(model, actionMap, episode)
                print(f"Loss: {np.mean(env.lossList)}")
                writer.writerow([episode, turnReward, np.mean(env.lossList), runTime])
                env.lossList.clear()
                env.reset()
                model.updateEps()
            
        
        env.onDisconnect()
        print("結束")
    except KeyboardInterrupt:
        env.onDisconnect()
        print("結束")


if (__name__ == "__main__"):
    train()