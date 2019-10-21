import numpy as np
# from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from random import shuffle
from balance_data import find_split

file_index,train_data=find_split()

# keys = np.sort(df[1].unique())
# keys = pd.get_dummies(keys)
# new_labels = []
# for i in df[1]:
	# for col in keys.columns:
		# if i == col:
			# new_labels.append(list(keys[col]))
# df[1] = new_labels

# SPLIT
train_data=np.array_split(train_data,file_index)
output=1


WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'pygta5-drivev0.1-{}-{}-{}-epochs.model'.format(LR, 'googlenet', EPOCHS)
PREV_MODEL = 'pygta5-drive-0.001-googlenet-30-epochs.model'

LOAD_MODEL = False

# wl = 0
# sl = 0
# al = 0
# dl = 0

# wal = 0
# wdl = 0
# sal = 0
# sdl = 0
# nkl = 0

# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]
# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]
# nk = [0,0,0,0,0,0,0,0,1]

model = googlenet(WIDTH, HEIGHT, 3, LR, output=output, model_name=MODEL_NAME)
#model = alexnet(WIDTH, HEIGHT, LR)

# train = train_data[:-round(0.2*len(train_data))]
# test = train_data[-round(0.2*len(train_data)):]

# X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
# Y = [i[1] for i in train]

# test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
# test_y = [i[1] for i in test]

# model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
	# snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
# print('SAVING MODEL!')
# model.save(MODEL_NAME)
				
if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
    

# iterates through the training files


for e in range(EPOCHS):
	#data_order = [i for i in range(1,file_index+1)]
	data_order = [i for i in range(0,file_index)]
	shuffle(data_order)
	for count,i in enumerate(data_order):
		
		try:
			#file_name = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data/training_data-{}.npy'.format(i)
			# full file info
			#train_data = np.load(file_name)
			data = train_data[i]
			
			print('training_data-{}.npy'.format(i+1),len(data))

	##            # [   [    [FRAMES], CHOICE   ]    ] 
	##            train_data = []
	##            current_frames = deque(maxlen=HM_FRAMES)
	##            
	##            for ds in data:
	##                screen, choice = ds
	##                gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
	##
	##
	##                current_frames.append(gray_screen)
	##                if len(current_frames) == HM_FRAMES:
	##                    train_data.append([list(current_frames),choice])


			# #
			# always validating unique data: 
			#shuffle(train_data)
			train = data[:-50]
			test = data[-50:]

			X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
			Y = [i[1] for i in train]
			Y = np.reshape(Y, (-1,1))

			test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
			test_y = [i[1] for i in test]
			test_y = np.reshape(test_y, (-1,1))

			model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
				snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)


			if count%10 == 0:
				print('SAVING MODEL!')
				model.save(MODEL_NAME)
					
		except Exception as e:
			print(str(e))
            
    








#

#tensorboard --logdir=foo:C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/log

