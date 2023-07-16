import pybullet as p 
from time import sleep
import os
chk = os.path.exists("./urdfs/myownhumanoid.urdf")
print("Path exists!", chk)
p.connect(p.GUI)
p.loadURDF("./urdfs/myownhumanoid.urdf") 
# p.loadURDF("./urdfs/racecar.urdf") 
sleep(3) 