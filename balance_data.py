# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
from sklearn.preprocessing import LabelEncoder

#FILE_I_END = 201

def balance_data(FILE_I_END,gps):
	training_data_full = []

	data_order = [i for i in range(1,FILE_I_END+1)]
	
	for count,i in enumerate(data_order):
		if gps:
			file_name = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_waypoint/training_data-{}.npy'.format(i)
		else:
			file_name = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_freedrive/training_data-{}.npy'.format(i)

		train_data = np.load(file_name)
		for data in train_data:
			training_data_full.append(data)
	
	df = pd.DataFrame(training_data_full)
	df[1]=df[1].apply(lambda x: ','.join(x))
	#onehot = OneHotEncoder().fit(df[1])
	le = LabelEncoder().fit(df[1])
	df[1] = le.transform(df[1])
	df = df.groupby(1).head(df[1].value_counts().tolist()[1])
	valuecounts = df[1].value_counts().tolist()
	valueindex = df[1].value_counts().index.tolist()
	for i in valueindex[-7:]:
		df = df[df[1] != i]
		
	training_data_full = df.values
	shuffle(training_data_full)
	
	if gps:
		np.save('training_data_balanced/waypoint/training_data.npy', training_data_full)
	else:
		np.save('training_data_balanced/freedrive/training_data.npy', training_data_full)
	# for count,i in enumerate(data_order):
		# try:
			# file_name = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_new/training_data-{}.npy'.format(i)
			# # full file info
			# train_data = np.load(file_name)
			# df = pd.DataFrame(train_data)
			# df[1] = LabelEncoder().fit_transform(df[1])
			# # print(df.head())
			# # print(Counter(df[1].apply(str)))

			# w = []
			# s = []
			# a = []
			# d = []
			# wa = []
			# wd = []
			# sa = []
			# sd = []
			# nk = []

			# shuffle(train_data)

			# for data in train_data:
				# img = data[0]
				# choice = data[1]

				# if choice == [1,0,0,0,0,0,0,0,0]:
					# w.append([img,choice])
				# elif choice == [0,1,0,0,0,0,0,0,0]:
					# s.append([img,choice])
				# elif choice == [0,0,1,0,0,0,0,0,0]:
					# a.append([img,choice])
				# elif choice == [0,0,0,1,0,0,0,0,0]:
					# d.append([img,choice])
				# elif choice == [0,0,0,0,1,0,0,0,0]:
					# wa.append([img,choice])
				# elif choice == [0,0,0,0,0,1,0,0,0]:
					# wd.append([img,choice])
				# elif choice == [0,0,0,0,0,0,1,0,0]:
					# sa.append([img,choice])
				# elif choice == [0,0,0,0,0,0,0,1,0]:
					# sd.append([img,choice])
				# elif choice == [0,0,0,0,0,0,0,0,1]:
					# nk.append([img,choice])
				# else:
					# print('no matches')


			# w = w[:len(a)][:len(d)]
			# a = a[:len(w)]
			# d = d[:len(w)]

			# final_data = w + s + a + d + wa + wd + sa + sd
			# training_data_full += final_data

		# except Exception as e:
			# print(str(e))

	if len(training_data_full) < 100000:
		print('Total data points: {}, please continue..'.format(len(training_data_full)))
	else:
		print("Good driving, let's train!")
	
	return training_data_full


def find_split():
	file_name = 'C:/Users/chris/Dropbox/MachineLearning/pygta5-master/pygta5-master/training_data_balanced/freedrive/training_data.npy'
	# full file info
	train_data = np.load(file_name)
	df = pd.DataFrame(train_data)
	valueindex = df[1].value_counts().index.tolist()
	for i in valueindex[-13:]:
		df = df[df[1] != i]
	# NORMALISE
	maximum = df[1].max(axis=0)
	minimum = df[1].min(axis=0)
	df[1] = [((i-minimum)/(maximum-minimum)) for i in df[1]]
	train_data = df.values
	split=False
	print('Beginning search...')
	while not split:
		lower = round(len(train_data)/500)
		upper = round(len(train_data)/500)+10
		for i in range(lower,upper+1):
			if len(train_data) % i == 0:
				file_index = int(i)
				split = True
		if not split:
			train_data=train_data[:-1]
			print(len(train_data))
			print('Trying again...')

	return file_index,train_data