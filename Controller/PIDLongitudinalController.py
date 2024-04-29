import carla
import numpy as np
from collections import deque
import math


# 縱向PID控制器
class PIDLongitudinalController:
    """
        param vehicel: carla Vehicel
        param K_P: Proportional term
        param K_D: Differential term
        param K_I: Integral term
        param dt: time differential in seconds
    """
    def __init__(self, vehicel: carla.Vehicle, maxThrottle = 0.75, maxBrake = 0.3, K_P = 1.0, K_I = 0.0, K_D = 0.0, dt = 0.03):
        self.vehicel = vehicel
        self.maxThrottle = maxThrottle
        self.maxBrake = maxBrake
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt
        self.errorBuffer = deque(maxlen = 10)
    
    # 回傳vehicel的速度 (km/h)
    def getVehicelSpeed(vehicel: carla.Vehicle):
        velocity = vehicel.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    
    def pidControl(self, currSpeed, targetSpeed):
        error = targetSpeed - currSpeed
        self.errorBuffer.append(error)
        if (len(self.errorBuffer) >= 2):
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0
        return np.clip((self.K_P * error) + (self.K_D * de) + (self.K_I * ie), -1.0, 1.0)
    
    """
        運算PID
        param targetSpeed: 目標速度 (km/h)
        param control: 最後結果回傳在control
        retrun carla vehicel的control以達到目標
    """
    def getControl(self, targetSpeed: float, control: carla.VehicleControl):
        currSpeed = PIDLongitudinalController.getVehicelSpeed(self.vehicel)
        accelration = self.pidControl(currSpeed, targetSpeed)
        if (accelration >= 0.0):
            control.throttle = min(accelration, self.maxThrottle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(accelration), self.maxBrake)