'''Training LSTM on NASA dataset'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
import os,csv,math
import numpy as np
import sklearn.model_selection, sklearn.preprocessing

def get_train_test_data(foldername, max_size,num_of_features=44):
	'''Pre-processing data with min-max normalization of data'''
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

		feature_maxs = np.maximum(np.amax(temp_arr,axis=0),feature_maxs)
		feature_mins = np.minimum(np.amin(temp_arr,axis=0),feature_mins)
		
		series.append(temp_arr) 
		sizes.append(temp_arr.shape[0])
	
	np.save("feature_maxs",feature_maxs)
        np.save("feature_mins",feature_mins)

	feature_maxs = np.load("feature_maxs.npy")
        feature_mins = np.load("feature_mins.npy")

	for ind in range(len(series)):
		#min-max normalization of series
		series[ind] = (series[ind] - feature_mins)/np.where((feature_maxs - feature_mins) != 0.0, (feature_maxs - feature_mins), 1.0)
	series = np.array(series)
	print series.shape
	train, test = sklearn.model_selection.train_test_split(series, train_size = 1500, random_state = 42)
	return train, test

num_features = 17
num_hidden = 15
output_size = num_features
train_set, test_set = get_train_test_data("40", max_size=3000, num_of_features=num_features)	
print train_set.shape, test_set.shape

#defining the model
lstm = nn.LSTM(num_features, num_hidden).cuda()
output_layer = nn.Linear(num_hidden, output_size).cuda()

loss_function = nn.MSELoss()
optimizer = optim.Adam([
                {'params': lstm.parameters()},
                {'params': output_layer.parameters()}
            ], lr = 0.0003)

for epoch in range(1000):  
    print epoch, 
    loss_val = 0
    for input_data in train_set:
        
        lstm.zero_grad()
	output_layer.zero_grad()

        
	hidden = (autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True), autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor), requires_grad=True))
	
	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data[:-1]), requires_grad=True)
        output, hidden = lstm(input_torchvar.view(len(input_data) - 1, 1, -1), hidden)

	output_2 = output_layer(output)
	
	loss = loss_function(output_2, autograd.Variable(torch.cuda.FloatTensor(input_data[1:])))
	loss_val += loss
        loss.backward()
        optimizer.step()

    print loss_val.data[0]/len(train_set)

    torch.save(lstm, "lstm_trained_model")
    torch.save(output_layer, "output_layer_trained_model")	


