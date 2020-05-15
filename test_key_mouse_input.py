# Installed
# import cv2
# import numpy as np
import time

# Project
# from grabscreen import grab_screen
import keys as k

# from getkeys import key_check

keys = k.Keys({})


for i in list(range(3))[::-1]:
    print(i + 1)
    time.sleep(1)


keys.directMouse(100, -100)
