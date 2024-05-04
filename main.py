import tensorflow as tf

from Agent.Agent import Agent
from Environment.Environment import Environment
from config import *
from NeuralNetwork.DQN import DQN

def enableGPU():
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
    try:
        enableGPU()
        agent = Agent()
        # # env = Environment(agent, ip="192.168.168.13")
        env = Environment(agent, ip="localhost")
        model = DQN(len(actionMap))
        env.runOneEpisode(model, actionMap)
        
        env.onDisconnect()
        print("結束")
    except KeyboardInterrupt:
        env.onDisconnect()
        print("結束")


if (__name__ == "__main__"):
    main()