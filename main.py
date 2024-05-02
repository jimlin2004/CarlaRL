import tensorflow as tf

from Agent.Agent import Agent
from Environment.Environment import Environment

# def showImage(carlaImg):
#     img = np.reshape(np.copy(carlaImg.raw_data), (carlaImg.height, carlaImg.width, 4))
#     cv2.imshow("Img", img)
#     cv2.waitKey(1)

# def main():
#     try:
#         client = carla.Client("192.168.168.15", 2000)
#         client.set_timeout(20.0)
#         world = client.get_world()
#         bpLib = world.get_blueprint_library()
#         spawnPoints = world.get_map().get_spawn_points()

#         vehicleBp = bpLib.find("vehicle.lincoln.mkz_2020")
#         vehicle = world.try_spawn_actor(vehicleBp, random.choice(spawnPoints))
#         vehicle.set_autopilot(True)
#         # agent = Agent(vehicle)
#         spectator = world.get_spectator()
        
#         cameraQueue = queue.Queue()
        
#         cameraBp = bpLib.find("sensor.camera.rgb")
#         cameraTransform = carla.Transform(carla.Location(x = 0.8, z = 1.4))
#         camera = world.spawn_actor(cameraBp, cameraTransform, attach_to=vehicle)
#         camera.listen(lambda image: cameraQueue.put(image))

#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.01
#         world.apply_settings(settings)

#         while (True):
#             # agent.setAccelerator(random.uniform(0.0, 1.0))
#             # agent.update()
            
#             world.tick()
#             showImage(cameraQueue.get())
#             transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), vehicle.get_transform().rotation)
#             spectator.set_transform(transform)
#     finally:
#         vehicle.destroy()
#         camera.destroy()
#         settings = world.get_settings()
#         settings.synchronous_mode = False
#         settings.fixed_delta_seconds = None
#         world.apply_settings(settings)
    
# if (__name__ == "__main__"):
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("結束")

def enableGPU():
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
    try:
        enableGPU()
        
        # agent = Agent()
        # # env = Environment(agent, ip="192.168.168.13")
        # env = Environment(agent, ip="localhost")
        # env.runOneEpisode()
        
        # env.onDisconnect()
        # print("結束")
    except KeyboardInterrupt:
        # env.onDisconnect()
        print("結束")


if (__name__ == "__main__"):
    main()