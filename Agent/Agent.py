import carla

# 設定油門 throttle
# val: [0.0, 1.0]
# 設定轉向 steer
# val: [-1.0, 1.0]
# 設定剎車brake
# val = [0.0, 1.0]

class Agent:
    # vehicle: Carla的車子
    def __init__(self):
        self.control = carla.VehicleControl()
        self.vehicle = None
        self.camera = None
        self.collisionSensor = None
        self.speedController = None

    def reset(self):
        self.vehicle.destroy()
        self.camera.destroy()
        self.collisionSensor.destroy()

    def update(self):
        self.vehicle.apply_control(self.control)