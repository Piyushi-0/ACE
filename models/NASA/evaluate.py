import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
import os,csv,math, sys
import numpy as np
import sklearn.model_selection, sklearn.preprocessing
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_train_test_data(foldername, max_size,num_of_features=44):
	series = []
	sizes = []
	feature_mins = [np.inf]*num_of_features
	feature_maxs = [-1*np.inf]*num_of_features
	for filename in os.listdir("./" + foldername + "/"):
		try:
		        f = open("./" + foldername + "/" + filename)
		except:
		        continue
		reader = csv.reader(f)

		lines = []
		for line in reader:
		        lines.append(line)

		temp_arr = np.array(lines[1:min(max_size + 1,len(lines))]).astype(float)

		series.append(temp_arr) 
		sizes.append(temp_arr.shape[0])
	

	feature_maxs = np.load("feature_maxs.npy")
        feature_mins = np.load("feature_mins.npy")

	for ind in range(len(series)):
		#min-max normalization of series
		series[ind] = (series[ind] - feature_mins)/np.where((feature_maxs - feature_mins) != 0.0, (feature_maxs - feature_mins), 1.0)
		#series[ind] = np.lib.pad(series[ind], ((0,max(0,max_size - series[ind].shape[0])),(0,0)), 'constant')
	series = np.array(series)
	#print series	
	print series.shape
	return series

def length(sequence):
	used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	return length

def get_weight_and_bias(in_size, out_size):
	n_avg = (in_size + out_size)/2.0
	weight = tf.truncated_normal([in_size, out_size], stddev=(1/math.sqrt(n_avg)))
        bias = tf.constant(0., shape=[out_size])
	return tf.Variable(weight), tf.Variable(bias)

def mse_step(a , b):
	diff = a - b
	d = diff.cpu().detach().numpy()
	d = np.absolute(d)**2
	n = d.shape[2]
	d = d.sum(axis = 2)/n
	return d

num_features = 17
num_hidden = 15
output_size = num_features
test_set = get_train_test_data(sys.argv[1], max_size=3000, num_of_features=num_features)	


lstm = torch.load("lstm_trained_model")
output_layer = torch.load("output_layer_trained_model") 

loss_function = nn.MSELoss()
loss_val = 0
mse = []
for input_data in test_set:
	
	lstm.zero_grad()
	output_layer.zero_grad()

	
	hidden = (autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True), autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True))

	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data[:-1]), requires_grad=True)
	output, hidden = lstm(input_torchvar.view(len(input_data) - 1, 1, -1), hidden)

	output_2 = output_layer(output)

	loss = loss_function(output_2, autograd.Variable(torch.cuda.FloatTensor(input_data[1:])).view(len(input_data) - 1, 1, -1))
	loss_val += loss
	mse_per_step = mse_step(output_2, autograd.Variable(torch.cuda.FloatTensor(input_data[1:])).view(len(input_data) - 1, 1, -1))
	mse.extend(mse_per_step)

#print loss_val.data[0]/len(test_set)
#print sum(mse)/len(mse)
X = np.arange(len(mse))
Y = mse
#anomaly_time_step = next(x[0] for x in enumerate(Y) if x[1] >=0.0508934557438)
anomaly_time_step = [x[0] for x in enumerate(Y) if x[1] >= 0.0508934557438]
print(anomaly_time_step)
plt.xlabel('Time steps')
plt.ylabel('MSE')
plt.plot(X, Y)
plt.plot(X,np.array([0.0508934557438]*X.shape[0]), color = 'red', label="threshold")
plt.show()
plt.savefig('MSE_step.png')

