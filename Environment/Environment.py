import carla
import random
import queue
import cv2
import numpy as np
import time
import copy
from scipy.spatial import KDTree

from Controller.PIDLongitudinalController import PIDLongitudinalController
from Agent.Agent import Agent
from NeuralNetwork.DQN import DQN
from agents.navigation.global_route_planner import GlobalRoutePlanner

def getCorrectYaw(x):
    return (((x % 360) + 360) % 360)

# carla image to opencv img
def carlaImg2CVMat(carlaImg: carla.Image) -> cv2.Mat:
    img = np.reshape(np.copy(carlaImg.raw_data), (carlaImg.height, carlaImg.width, 4))
    return img 

def resizeImg(img: cv2.Mat, shape):
    return cv2.resize(img, shape)

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
        self.topImageQueue = queue.Queue()
        self.collisionsQueue = queue.Queue()
        # self.laneQueue = queue.Queue()
        
        self.maxDisFormWaypoint = 2.0
        
        self.trainCnt = 0
        self.trainFreq = 1
        
        self.totalReward = 0
        self.saveFreq = 100
        
        self.lossList = []
        
        self.waypointsMp = {}
        self.kdTreeMp = {}
        self.processWaypoints()

    def setAgentAndSensor(self):
        vehicleBp = self.bpLib.find("vehicle.lincoln.mkz_2020")
        selectedSpawnPoints = random.choice(self.spawnPoints)
        self.agent.vehicle = self.world.try_spawn_actor(vehicleBp, selectedSpawnPoints)
        
        # agent的camera設定
        self.imageQueue.queue.clear()
        cameraBp = self.bpLib.find("sensor.camera.rgb")
        cameraTransform = carla.Transform(carla.Location(x = 0.8, z = 1.4))
        self.agent.camera = self.world.spawn_actor(cameraBp, cameraTransform, attach_to = self.agent.vehicle)
        self.agent.camera.listen(lambda image: self.imageQueue.put(image))
        
        self.topImageQueue.queue.clear()
        topCameraTransform = carla.Transform(carla.Location(x = 0, z = 10), carla.Rotation(pitch=-90.0))
        topCameraBp = self.bpLib.find("sensor.camera.rgb")
        self.agent.topCamera = self.world.spawn_actor(topCameraBp, topCameraTransform, attach_to = self.agent.vehicle)
        self.agent.topCamera.listen(lambda image: self.topImageQueue.put(image))
        
        # agent的collision sensor設定
        self.collisionsQueue.queue.clear()
        collisionSensorBp = self.bpLib.find("sensor.other.collision")
        self.agent.collisionSensor = self.world.spawn_actor(collisionSensorBp, carla.Transform(), attach_to = self.agent.vehicle)
        self.agent.collisionSensor.listen(lambda event: self.collisionsQueue.put(event))
        
        # agent的PID controller設定
        self.agent.speedController = PIDLongitudinalController(self.agent.vehicle)
        
        return selectedSpawnPoints

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
    
    def showOpenCVWindow(self, img: cv2.Mat, windowName: str):
        cv2.imshow(windowName, img)
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
        
    def processWaypoints(self):
        allWaypoints = self.world.get_map().generate_waypoints(0.2)
        size = len(allWaypoints)
        for i in range(0, size):
            if (allWaypoints[i].lane_id not in self.waypointsMp):
                self.waypointsMp[allWaypoints[i].lane_id] = [allWaypoints[i]]
            else:
                self.waypointsMp[allWaypoints[i].lane_id].append(allWaypoints[i])
        for key in self.waypointsMp.keys():
            wpLocations = []
            size = len(self.waypointsMp[key])
            wps = copy.copy(self.waypointsMp[key])
            for i in range(0, size):
                wpLocations.append([wps[i].transform.location.x, wps[i].transform.location.y])
            self.kdTreeMp[key] = KDTree(wpLocations)
        
    def runAutopilotEnv(self):
        spawnPoint = self.setAgentAndSensor()
        
        originWp = self.world.get_map().get_waypoint(self.agent.vehicle.get_location(), project_to_road=True)
        
        # wps = copy.copy(self.waypointsMp[originWp.lane_id])
        kdTree = copy.copy(self.kdTreeMp[originWp.lane_id])
        
        self.setWorldSync()
        self.agent.vehicle.set_autopilot(True)
        
        spectator = self.world.get_spectator()
            
        index = 0
        while (True):
            self.agent.speedController.getControl(20, self.agent.control)
            self.agent.update()
            
            self.world.tick()
            carlaImg = self.imageQueue.get()
            img_cv = carlaImg2CVMat(carlaImg)
            if (self.isOpenOpenCVWindow):
                self.showOpenCVWindow(img_cv, "camera")
            img = resizeImg(img_cv, (128, 128))
            nextState = processImgToAI(img)
            
            vehicleLocation = self.agent.vehicle.get_location()
            print(kdTree.query([vehicleLocation.x, vehicleLocation.y]))
            # isCollision = 0 if (self.detectCollision() == None) else 1
            
            # # transform = carla.Transform(self.agent.vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), self.agent.vehicle.get_transform().rotation)
            # transform = wps[index].transform
            # transform.location.z = 2.5
            # spectator.set_transform(transform)
            
            # index = (index + 1) % len(wps)
            # agentLocation = self.agent.vehicle.get_location()
            # isFall = 0 if (agentLocation.z > -5) else 1
    
    def runOneEpisode(self, model: DQN, actionMap, episode: int, training = True):
        # 當輪reward總和
        turnReward = 0
        
        self.setAgentAndSensor()
        self.setWorldSync()
        
        # spectator = self.world.get_spectator()
        # 拿到當前state(img)
        
        # 讓車子先著地
        prevZ = self.agent.vehicle.get_location().z
        while (1):
            self.world.tick()
            currZ = self.agent.vehicle.get_location().z
            carlaTopimg = self.topImageQueue.get()
            carlaImg = self.imageQueue.get()
            img_cv = carlaImg2CVMat(carlaImg)
            img = resizeImg(img_cv, (128, 128))
            nextState = processImgToAI(img)
            
            if (currZ - prevZ >= 0):
                break
            prevZ = currZ
            
        originWaypoint = self.world.get_map().get_waypoint(self.agent.vehicle.get_location(), project_to_road=True)
        
        wps = copy.copy(self.waypointsMp[originWaypoint.lane_id])
        kdTree = copy.copy(self.kdTreeMp[originWaypoint.lane_id])
        # allWaypoins = originWaypoint.next_until_lane_end(0.2)
        startTime = time.time()
        currTime = 0
        # self.agent.vehicle.set_autopilot(True)
        while (True):
            self.trainCnt += 1
            state = nextState
            
            takenAction = model.selectAction(state, training)
            action = actionMap[takenAction]

            self.agent.speedController.getControl(20, self.agent.control)
            self.agent.control.steer = action
            
            self.agent.update()
            
            self.world.tick()
            
            carlaTopImg = self.topImageQueue.get()
            topImg_cv = carlaImg2CVMat(carlaTopImg)
            if (self.isOpenOpenCVWindow):
                topImgHeight, topImgWidth = topImg_cv.shape[:2]
                topImgAspectRatio = topImgWidth / topImgHeight
                topImg_cv_resized = resizeImg(topImg_cv, (int(512 * topImgAspectRatio), 512))
                self.showOpenCVWindow(topImg_cv_resized, "topCamera")
            
            carlaImg = self.imageQueue.get()
            img_cv = carlaImg2CVMat(carlaImg)
            if (self.isOpenOpenCVWindow):
                img_cv_resized = resizeImg(img_cv, (512, 512))
                self.showOpenCVWindow(img_cv_resized, "camera")
            img = resizeImg(img_cv, (128, 128))
            nextState = processImgToAI(img)
            
            isCollision = 0 if (self.detectCollision() == None) else 1
            
            agentLocation = self.agent.vehicle.get_location()
            isFall = 0 if (agentLocation.z > -5) else 1
            
            closestPoint = wps[kdTree.query([agentLocation.x, agentLocation.y])[1]]
            
            # cosYawDiff, dis = getRewardArgs(self.agent.vehicle, waypoint)
            cosYawDiff, dis = getRewardArgs(self.agent.vehicle, closestPoint)
            reward = getRewardValue(cosYawDiff, dis, isCollision, isFall)
            turnReward += reward
            
            done = (isCollision == 1 or isFall == 1 or (dis > self.maxDisFormWaypoint))
            
            # 道路有終點
            if (self.isTheWayWillEnd):
                if (closestPoint == wps[-1]):
                    done = True
            
            if (training):
                model.replayBuffer.push(state, takenAction, nextState, reward, done)

            if (training and (self.trainCnt >= self.batchSize) and (self.trainCnt % self.trainFreq == 0)):
                loss = model.train()
                self.lossList.append(loss)
            
            currTime = time.time()
            # 超過60秒
            if (currTime - startTime >= 60):
                done = True
            if (done):
                break

        print(f"[episode {episode}] Reward: {turnReward}, time: {currTime - startTime}")
        self.totalReward += turnReward
        if (training and (episode % self.saveFreq == 0)):
            saveModel(model, episode)
            self.log()
        return turnReward, currTime - startTime