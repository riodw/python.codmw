# train_model.py

import numpy as np
from alexnet import alexnet
WIDTH = 96
HEIGHT = 58
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pycodmw-atv-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 22
# for i in range(EPOCHS):
#     for i in range(1,hm_data+1):
train_data = np.load('training_data-shuffled.npy', allow_pickle=True)

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH ,HEIGHT, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)



# py -m tensorboard.main --logdir=/Users/riodw/projects/python.codmw/log
