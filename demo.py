import tensorflow as tf
import numpy as np
import csv
import datetime

from Agent.Agent import Agent
from Environment.Environment import Environment
from config import *
from NeuralNetwork.DQN import DQN
from NeuralNetwork.DDQN import DDQN

def enableGPU():
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

def demo():
    try:
        enableGPU()
        agent = Agent()
        # # env = Environment(agent, ip="192.168.168.13")
        isTheWayWillEnd = False
        env = Environment(agent, BatchSize, isTheWayWillEnd, ip="localhost")
        model = DQN(len(actionMap), lr = LearningRate, batchSize=BatchSize)
        # model = DDQN(len(actionMap), lr = LearningRate, batchSize=BatchSize)
        
        # newEps = max(model.eps * (model.epsilonDecay ** 400), model.endEpsilon)
        # model.setEps(newEps)
        model.loadWeight("20240506_SGD_DQN_simpleStraight/weights/weight_episode500")
        
        # with open(f"./demoHistroy/demoHistory_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv", "w", newline="") as csvfile:
            # writer = csv.writer(csvfile)
        for episode in range(1, 1001):
            print(f"Current eps: {model.eps}")
            turnReward, runTime = env.runOneEpisode(model, actionMap, episode, False)
            # writer.writerow([episode, turnReward, runTime])
            env.lossList.clear()
            env.reset()
            model.updateEps()
                
        env.onDisconnect()
        print("結束")
    except KeyboardInterrupt:
        env.onDisconnect()
        print("結束")


if (__name__ == "__main__"):
    demo()