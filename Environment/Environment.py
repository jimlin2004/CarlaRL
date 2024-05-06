import carla
import random
import queue
import cv2
import numpy as np
import time

from Controller.PIDLongitudinalController import PIDLongitudinalController
from Agent.Agent import Agent
from NeuralNetwork.DQN import DQN

def getCorrectYaw(x):
    return (((x % 360) + 360) % 360)

def processImg(carlaImg: carla.Image, shape) -> cv2.Mat:
    img = np.reshape(np.copy(carlaImg.raw_data), (carlaImg.height, carlaImg.width, 4))
    img = cv2.resize(img, shape)
    return img 

def processImgToAI(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    imgData = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    return imgData

# 得到[與waypoint的角度差, 與waypoint的距離, collision]
def getRewardArgs(vehicle: carla.Vehicle, waypoint: carla.Waypoint):
    vehicleLocation = vehicle.get_location()
    xWaypoint = waypoint.transform.location.x
    yWaypoint = waypoint.transform.location.y
    xVehicle = vehicleLocation.x
    yVehicle = vehicleLocation.y
    
    wpArray = np.array([xWaypoint, yWaypoint])
    vehicleArray = np.array([xVehicle, yVehicle])
    
    dis = np.linalg.norm(wpArray - vehicleArray)
    
    vehicleYaw = getCorrectYaw(vehicle.get_transform().rotation.yaw)
    wpYaw = getCorrectYaw(waypoint.transform.rotation.yaw)
    
    cosYawDiff = np.cos((vehicleYaw - wpYaw) * np.pi / 180.0)
    return cosYawDiff, dis

def getRewardValue(cosYawDiff: float, dis: float, isCollision: int, isFall: int, lambda1 = 1, lambda2 = 1, lambda3 = 10, lambda4 = 10):
    return (lambda1 * cosYawDiff) - lambda2 * dis - (lambda3 * isCollision) - (lambda4 * isFall)

def saveModel(model: DQN, episode: int):
    model.Q.save_weights(f"./weights/weight_episode{episode}")

class Environment:
    def __init__(
        self, 
        agent: Agent,
        batchSize,
        isTheWayWillEnd,
        isOpenOpenCVWindow = True,
        ip = "localhost",
        port = 2000,
        timeout = 20.0,
        fps = 30
    ):
        self.agent = agent
        self.batchSize = batchSize
        self.isTheWayWillEnd = isTheWayWillEnd
        self.isOpenOpenCVWindow = isOpenOpenCVWindow
        self.fps = fps
        
        self.client = carla.Client(ip, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.bpLib = self.world.get_blueprint_library()
        self.spawnPoints = self.world.get_map().get_spawn_points()
        self.imageQueue = queue.Queue()
        self.collisionsQueue = queue.Queue()
        # self.laneQueue = queue.Queue()
        
        self.maxDisFormWaypoint = 2.0
        
        self.trainCnt = 0
        self.trainFreq = 1
        
        self.totalReward = 0
        self.saveFreq = 100
        
        self.lossList = []

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
        
        # laneSensor = self.bpLib.find("sensor.other.lane_invasion")
        # self.agent.laneSensor = self.world.spawn_actor(laneSensor, carla.Transform(), attach_to=self.agent.vehicle)
        # self.agent.laneSensor.listen(lambda event: self.laneQueue.put(event))
        
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
            return self.collisionsQueue.get(block=False)
        except queue.Empty:
            return None
        
    # def detectLaneInversion(self):
    #     try:
    #         return self.laneQueue.get(block=False)
    #     except queue.Empty:
    #         return None
    
    def showOpenCVWindow(self, img: cv2.Mat):
        cv2.imshow("Img", img)
        cv2.waitKey(1)

    # 設定world用sync mode
    def setWorldSync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        self.world.apply_settings(settings)

    def log(self):
        print(f"[AVG] Reward: {self.totalReward / self.saveFreq}")
        self.totalReward = 0
    
    def runOneEpisode(self, model: DQN, actionMap, episode: int):
        # 當輪reward總和
        turnReward = 0
        
        self.setAgentAndSensor()
        
        self.setWorldSync()
        
        # spectator = self.world.get_spectator()
        # 拿到當前state(img)
        
        # self.agent.vehicle.set_autopilot(True)
        # 讓車子先著地
        prevZ = self.agent.vehicle.get_location().z
        while (1):
            self.world.tick()
            currZ = self.agent.vehicle.get_location().z
            carlaImg = self.imageQueue.get()
            img = processImg(carlaImg, (128, 128))
            nextState = processImgToAI(img)
            
            if (currZ - prevZ >= 0):
                break
            prevZ = currZ
            
        originWaypoint = self.world.get_map().get_waypoint(self.agent.vehicle.get_location(), project_to_road=True)
        allWaypoins = originWaypoint.next_until_lane_end(0.2)
        startTime = time.time()
        currTime = 0
        
        while (True):
            self.trainCnt += 1
            state = nextState
            
            takenAction = model.selectAction(state)
            action = actionMap[takenAction]

            # # self.agent.speedController.getControl(action[1], self.agent.control)
            self.agent.speedController.getControl(20, self.agent.control)
            # self.agent.control.steer = actionMap[5]
            self.agent.control.steer = action
            
            self.agent.update()
            
            self.world.tick()
            carlaImg = self.imageQueue.get()
            img = processImg(carlaImg, (128, 128))
            if (self.isOpenOpenCVWindow):
                self.showOpenCVWindow(img)
            nextState = processImgToAI(img)
            
            # isLaneInvert = 0 if (self.detectLaneInversion() == None) else 1
            isCollision = 0 if (self.detectCollision() == None) else 1
            
            # transform = carla.Transform(self.agent.vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), self.agent.vehicle.get_transform().rotation)
            # spectator.set_transform(transform)
            agentLocation = self.agent.vehicle.get_location()
            isFall = 0 if (agentLocation.z > -5) else 1
            
            # waypoint = self.world.get_map().get_waypoint(agentLocation, project_to_road=True)
            currClosestDiff = 0x3f3f3f3f
            closestPoint = None
            for wp in allWaypoins:
                dis = agentLocation.distance(wp.transform.location)
                if (dis < currClosestDiff):
                    currClosestDiff = dis
                    closestPoint = wp
            
            # cosYawDiff, dis = getRewardArgs(self.agent.vehicle, waypoint)
            cosYawDiff, dis = getRewardArgs(self.agent.vehicle, closestPoint)
            reward = getRewardValue(cosYawDiff, dis, isCollision, isFall)
            turnReward += reward
            
            done = (isCollision == 1 or isFall == 1 or (dis > self.maxDisFormWaypoint))
            
            # 道路有終點
            if (self.isTheWayWillEnd):
                if (closestPoint == allWaypoins[-1]):
                    done = True
            model.replayBuffer.push(state, takenAction, nextState, reward, done)
            
            if ((self.trainCnt >= self.batchSize) and (self.trainCnt % self.trainFreq == 0)):
                loss = model.train()
                self.lossList.append(loss)
            
            currTime = time.time()
            # 超過30秒
            if (currTime - startTime >= 30):
                done = True
            if (done):
                break

        print(f"[episode {episode}] Reward: {turnReward}, time: {currTime - startTime}")
        self.totalReward += turnReward
        if (episode % self.saveFreq == 0):
            saveModel(model, episode)
            self.log()
        return turnReward, currTime - startTime