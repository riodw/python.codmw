import os
import time
# Installed
import numpy as np
from PIL import ImageGrab
import cv2
# Project
from directkeys import PressKey, W, A, S, D
from grabscreen import grab_screen
from getkeys import key_check

# \Users\riodw\projects\python.codmw

def keys_to_output(keys):
    # [A, W, D]
    output = [0, 0, 0]
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    
    return output

file_name = 'training_data.npy'

if False and os.path.isfile(file_name):
    print('File exists, loading previous data')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh')
    training_data = []




def main():
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()

    paused = False

    while True:

        keys = key_check()

        if not paused:
            # 800x600 windowed mode for Call of Duty Modern Warzone, at the top left position of main screen.
            # 40 px accounts for title bar. 
            # screen =  np.array(ImageGrab.grab(bbox=(0, 40, 800, 530)))
            screen =  grab_screen(region=(0, 40, 962, 579))
            # convert to gray scale
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            # resize to smaller screen
            cv2.imshow('window2', screen)
            screen = cv2.resize(screen, (96, 58))
            # screen preview
            # output
            output = keys_to_output(keys)
            training_data.append([screen, output])


            print('Frame took {} seconds'.format(time.time()-last_time))

            if len(training_data) % 500 == 0:
                print(len(training_data))
                np.save(file_name, training_data)
        
        last_time = time.time()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                time.sleep(1)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()