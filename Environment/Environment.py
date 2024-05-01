import carla
import random
import queue
import cv2
import numpy as np

from Controller.PIDLongitudinalController import PIDLongitudinalController
from Agent.Agent import Agent

class Environment:
    def __init__(
        self, 
        agent: Agent,
        isOpenOpenCVWindow = True,
        ip = "localhost",
        port = 2000,
        timeout = 20.0
    ):
        self.agent = agent
        self.isOpenOpenCVWindow = isOpenOpenCVWindow
        self.client = carla.Client(ip, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.bpLib = self.world.get_blueprint_library()
        self.spawnPoints = self.world.get_map().get_spawn_points()
        self.imageQueue = queue.Queue()
        self.collisionsQueue = queue.Queue()

    def setAgentAndSensor(self):
        vehicleBp = self.bpLib.find("vehicle.lincoln.mkz_2020")
        self.agent.vehicle = self.world.try_spawn_actor(vehicleBp, random.choice(self.spawnPoints))
        
        # agent的camera設定
        self.imageQueue.queue.clear()
        cameraBp = self.bpLib.find("sensor.camera.rgb")
        cameraTransform = carla.Transform(carla.Location(x = 0.8, z = 1.4))
        self.agent.camera = self.world.spawn_actor(cameraBp, cameraTransform, attach_to = self.agent.vehicle)
        self.agent.camera.listen(lambda image: self.imageQueue.put(image))
        
        # agent的collision sensor設定
        self.collisionsQueue.queue.clear()
        collisionSensorBp = self.bpLib.find("sensor.other.collision")
        self.agent.collisionSensor = self.world.spawn_actor(collisionSensorBp, carla.Transform(), attach_to = self.agent.vehicle)
        self.agent.collisionSensor.listen(lambda event: self.collisionsQueue.put(event))
        
        # agent的PID controller設定
        self.agent.speedController = PIDLongitudinalController(self.agent.vehicle)

    def reset(self):
        self.agent.reset()
        
    def onDisconnect(self):
        self.reset()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        
    def detectCollision(self):
        try:
            print(self.collisionsQueue.get(block=False))
        except queue.Empty:
            return
    
    def showOpenCVWindow(self, carlaImg: carla.Image):
        img = np.reshape(np.copy(carlaImg.raw_data), (carlaImg.height, carlaImg.width, 4))
        img = cv2.resize(img, (320, 320))
        cv2.imshow("Img", img)
        cv2.waitKey(1)
        
    def runOneEpisode(self):
        self.setAgentAndSensor()
        # 設定world用sync mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01
        self.world.apply_settings(settings)
        
        spectator = self.world.get_spectator()
        
        while (True):
            self.agent.speedController.getControl(40.0, self.agent.control)
            self.agent.update()

            self.world.tick()
            if (self.isOpenOpenCVWindow):
                self.showOpenCVWindow(self.imageQueue.get())
            self.detectCollision()
            transform = carla.Transform(self.agent.vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), self.agent.vehicle.get_transform().rotation)
            spectator.set_transform(transform)