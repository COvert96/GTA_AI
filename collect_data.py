import numpy as np
from grabscreen import grab_screen
import cv2
#from PIL import ImageGrab
import time
from getkeys import key_check
import os
from balance_data import balance_data
import win32gui
import ctypes
import pandas as pd

# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]
# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]
# nk = [0,0,0,0,0,0,0,0,1]
# Y = np.array(['w','s','a','d','h','wa','wd','wh','sa','sd','sh','ha','hd'])
# df = pd.get_dummies(Y)


starting_value_waypoint = 1
starting_value_fd = 1
ctypes.windll.user32.SetProcessDPIAware()
while True:
	file_name_waypoint = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_waypoint/training_data-{}.npy'.format(starting_value_waypoint)
	file_name_fd = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_freedrive/training_data-{}.npy'.format(starting_value_fd)
	
	if os.path.isfile(file_name_waypoint):
		print('File exists, moving along',starting_value_waypoint)
		starting_value_waypoint += 1
	
	elif os.path.isfile(file_name_fd):
		print('File exists, moving along',starting_value_fd)
		starting_value_fd += 1
	
	# elif not os.path.isfile(file_name_waypoint):
		# # print('Waypoint files finished',starting_value_waypoint)
		
		# pass
		
	# elif not os.path.isfile(file_name_fd):
		# # print('Freedrive files finished',starting_value_fd)
		
		# pass
	
	elif (not os.path.isfile(file_name_waypoint) and not os.path.isfile(file_name_fd)):
		print('All files considered, starting fresh! Waypoint: {}, Freedrive: {}'.format(starting_value_waypoint,starting_value_fd))
		
		break
	

def keys_to_output(keys):
	'''
	Convert keys to a ...multi-hot... array
	 0  1  2  3  4   5   6   7    8
	[W, S, A, D, H, WA, WD, WH, SA, SD, SH, HA, HD, NOKEY] boolean values.
	'''
	output = [0,0,0,0,0,0,0,0,0,0,0,0]

	if 'W' in keys and 'A' in keys:
		output = list(df['wa'])
	elif 'W' in keys and 'D' in keys:
		output = list(df['wd'])
	elif 'S' in keys and 'A' in keys:
		output = list(df['sa'])
	elif 'S' in keys and 'D' in keys:
		output = list(df['sd'])
	elif ' ' in keys and 'A' in keys:
		output = list(df['ha'])
	elif ' ' in keys and 'D' in keys:
		output = list(df['hd'])
	elif ' ' in keys and 'S' in keys:
		output = list(df['sh'])
	elif ' ' in keys and 'W' in keys:
		output = list(df['wa'])
	elif 'W' in keys:
		output = list(df['w'])
	elif 'S' in keys:
		output = list(df['s'])
	elif 'A' in keys:
		output = list(df['a'])
	elif 'D' in keys:
		output = list(df['d'])
	elif ' ' in keys:
		output = list(df['h'])
	else:
		output = output #no key pressed
	return output


def main(file_name_waypoint,file_name_fd, starting_value_waypoint,starting_value_fd):
	file_name_waypoint = file_name_waypoint
	file_name_fd = file_name_fd
	starting_value_waypoint = starting_value_waypoint
	starting_value_fd = starting_value_fd
	training_data = []
	training_count = 0
	training_data_fd = []
	training_count_fd = 0
	
	hwnd = win32gui.FindWindow(None, 'Grand Theft Auto V')
	bbox = list(win32gui.GetWindowRect(hwnd))
	bbox[0] += 5
	bbox[1] += 50
	bbox[2] -= 5
	bbox[3] -= 5
	bbox = tuple(bbox)
	
	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)

	screen = grab_screen(region=bbox)
	res = list(screen.shape)[:2]
	map_screen = screen[round(res[0]-0.2*res[0]):res[0],0:round(res[0]-0.77*res[0])]
	
	# define the list of boundaries
	boundary = ([140,0,200], [255,130,255])
	lower = np.array(boundary[0],dtype='uint8')
	upper = np.array(boundary[1],dtype='uint8')

	#last_time = time.time()
	paused = False
	gps = (True if np.sum(cv2.inRange(map_screen, lower, upper))>100 else False)
	print('STARTING!!!')
	while(True):

		if not paused:
			#last_time = time.time()
			#screen = np.array(ImageGrab.grab(bbox=bbox))
			screen = grab_screen(region=bbox)
			# keys = key_check()
			output = key_check()
			
			# resize to something a bit more acceptable for a CNN
			screen = cv2.resize(screen, (160,120))
			# run a color convert:
			screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)


			##            cv2.imshow('window',cv2.resize(screen,(640,360)))
			##            if cv2.waitKey(25) & 0xFF == ord('q'):
			##                cv2.destroyAllWindows()
			##                break
			if gps:
				training_data.append([screen,output])
				training_count += 1
				if len(training_data) % 100 == 0:
					print(len(training_data))
					
					if len(training_data) == 500:
						np.save(file_name_waypoint,training_data)
						print('SAVED GPS {}, Session total: {}'.format(starting_value_waypoint,training_count))
						training_data = []
						starting_value_waypoint += 1
						file_name_waypoint = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_waypoint/training_data-{}.npy'.format(starting_value_waypoint)
			else:
				training_data_fd.append([screen,output])
				training_count_fd += 1
				if len(training_data_fd) % 100 == 0:
					print(len(training_data_fd))
					
					if len(training_data_fd) == 500:
						np.save(file_name_fd,training_data_fd)
						print('SAVED FREEDRIVE {}, Session total: {}'.format(starting_value_fd,training_count_fd))
						training_data_fd = []
						starting_value_fd += 1
						file_name_fd = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_freedrive/training_data-{}.npy'.format(starting_value_fd)
			#print('loop took {} seconds'.format(time.time()-last_time))			
					
		keys = key_check()

		if 'B' in keys:
			if paused:
				if gps:
					balance_data(starting_value_waypoint-1,gps)
				else:
					balance_data(starting_value_fd-1,gps)

		if 'P' in keys:
			if paused:
				paused = False
				print('unpaused!')
				time.sleep(1)
				screen = grab_screen(region=bbox)
				map_screen = screen[round(res[0]-0.2*res[0]):res[0],0:round(res[0]-0.77*res[0])]
				# find the colors within the specified boundaries and apply
				# the mask
				gps = (True if np.sum(cv2.inRange(map_screen, lower, upper))>100 else False)
				print(gps)
			else:
				print('Pausing!')
				paused = True
				if gps:
					training_count-=len(training_data)
					training_data=[]
				else:
					training_count_fd -= len(training_data_fd)
					training_data_fd = []
					
				time.sleep(1)

main(file_name_waypoint,file_name_fd, starting_value_waypoint,starting_value_fd)
