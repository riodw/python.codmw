# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data-shuffled.npy', allow_pickle=True)
print(len(train_data))

# REVIEW TRAINING DATA
if True:
    i = 0
    for data in train_data:
        i = i + 1
        img = data[0]
        print(img.shape)
        choice = data[1]
        cv2.imshow('window', img)
        # print(i, choice)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if False:
    df = pd.DataFrame(train_data)
    print(df.head())
    print(Counter(df[1].apply(str)))

    lefts = []
    rights = []
    forwards = []

    shuffle(train_data)

    for data in train_data:
        img = data[0]
        choice = data[1]

        if choice == [1,0,0]:
            lefts.append([img,choice])
        elif choice == [0,1,0]:
            forwards.append([img,choice])
        elif choice == [0,0,1]:
            rights.append([img,choice])
        else:
            print('no matches')


    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    final_data = forwards + lefts + rights
    shuffle(final_data)
    print(len(final_data))

    np.save('training_data-shuffled.npy', final_data)
