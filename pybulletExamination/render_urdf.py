import pybullet as p 
from time import sleep
import os
chk = os.path.exists("./urdfs/racecar.urdf")
print("Path exists!", chk)
p.connect(p.GUI)
p.loadURDF("./urdfs/myown.urdf") 
# p.loadURDF("./urdfs/racecar.urdf") 
sleep(3) 