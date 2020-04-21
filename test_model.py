# test_model.py

import time
# Installed
import numpy as np
import cv2
# Project
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 96
HEIGHT = 58
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pycodmw-atv-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.0

def straight():
    # if random.randrange(3) == 2:
    #    ReleaseKey(W)
    # else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    # PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    # time.sleep(t_time)
    # ReleaseKey(A)

def right():
    # PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    # time.sleep(t_time)
    # ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False

    while(True):
        
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0, 40, 962, 579))
            print('Frame loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
