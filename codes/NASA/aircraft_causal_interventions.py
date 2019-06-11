import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
import os,csv,math, sys, pickle
import numpy as np
import sklearn.model_selection, sklearn.preprocessing

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

def interventions(lstm, output_layer, feature_index, output_index, data):
	#all parameters are between 0-1
	sampling_of_causal_feature = np.linspace(0.0,1.0,100)
	flight_sequences_do_x = {}
	for sample in sampling_of_causal_feature:
		sequence = []
		for flight_sequence in data:
	
			lstm.zero_grad()
			output_layer.zero_grad()

			# initialize the hidden state.
			hidden = (autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True), autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True))

			#future doesn't affect past and so only unfold the network till feature index, whose causal effect we are interested in estimating
			if flight_sequence.shape[0] < feature_index[0]:
				continue

			input_data = flight_sequence[:feature_index[0],:]
	
			input_torchvar = autograd.Variable(torch.FloatTensor(input_data), requires_grad=True) #assume at same timestep features are independent
			
			output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)

			output_2 = output_layer(output)	
		
			output_buffer = np.array(input_data)
			
			
			current_timestep_input = torch.FloatTensor(flight_sequence[feature_index[0],:])
			current_timestep_input[feature_index[2]] = sample
			
			output_buffer = np.vstack((output_buffer,current_timestep_input.numpy()))

			for timestep in range(feature_index[0], effect[0]):
				output, hidden = lstm(autograd.Variable(current_timestep_input, requires_grad=True).view(1, 1, -1), hidden)
				current_timestep_input = torch.FloatTensor(output_layer(output).data)	
				
				output_buffer = np.vstack((output_buffer,current_timestep_input.numpy().reshape(1,-1)))	
			sequence.append(output_buffer)

		flight_sequences_do_x[sample] = np.array(sequence)

	pickle_out = open("flight_sequences_do_x","wb")
	pickle.dump(flight_sequences_do_x, pickle_out)
	pickle_out.close()


def causal_strength(lstm, output_layer, feature_index, output_index):
	pickle_in = open("flight_sequences_do_x","rb")
	flight_sequences_do_x = pickle.load(pickle_in)
	do_x_expectation_series = []
	index = 1	
	for key in flight_sequences_do_x.keys():	#get Expectation with do=x for every sample
		flight_sequence = flight_sequences_do_x[key]
		
		means = np.mean(flight_sequence, axis=0)
		covs = np.cov(flight_sequence.reshape(-1,int(flight_sequence.shape[1])*int(flight_sequence.shape[2])),rowvar=False)
		
		hidden = (autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True), autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.FloatTensor), requires_grad=True))
		input_torchvar = autograd.Variable(torch.FloatTensor(means), requires_grad=True)
		output, hidden = lstm(input_torchvar.view(len(means), 1, -1), hidden)

		output_2 = output_layer(output)	
	
		grad_mask = torch.zeros(output_2.size()).type(torch.FloatTensor)

		grad_mask[output_index] = 1.0
	
		first_grads = torch.autograd.grad(output_2, input_torchvar, grad_outputs=grad_mask, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=False)
		do_x_expectation = float(output_2[output_index])

		first_grad_shape = first_grads[0].data.size()
		
		for i in range(first_grad_shape[0]):
			
			for j in range(first_grad_shape[1]):
				grad_mask = torch.zeros(first_grad_shape[0],first_grad_shape[1])
				grad_mask[i][j] = 1.0
				second_grads = torch.autograd.grad(first_grads, input_torchvar, grad_outputs=grad_mask, retain_graph=True, create_graph=False, only_inputs=True, allow_unused=False)  

				second_grads_array = np.array(second_grads[0].data)
				do_x_expectation += 0.5*np.sum(second_grads_array*covs[i*first_grad_shape[1] + j].reshape(first_grad_shape))
		print index, do_x_expectation	
		do_x_expectation_series.append(do_x_expectation)
		index += 1
		
	with open('expectation_do_x_for_cause_'+str(feature_index[0]) + "-"+str(feature_index[2]) + "_and_effect_" + str(output_index[0]) + "-"+str(output_index[2]), 'wb') as fp:
    		pickle.dump(do_x_expectation_series, fp)
	return	
	
num_features = 17
num_hidden = 15
output_size = num_features

test_set = np.array(get_train_test_data(sys.argv[1], max_size=3000, num_of_features=num_features))	

lstm = torch.load("lstm_trained_model").cpu()
output_layer = torch.load("output_layer_trained_model").cpu()

cov = 0#np.dot(test_set[:,:-1,:].reshape((-1,2999*17)).T,test_set[:,:-1,:].reshape((-1,2999*17)))/float(test_set.shape[0] - 1)

headers_file = open("headers.csv","r").read()
headers = headers_file.split(",")

effect = (100,0,14)

print "effect is", headers[effect[2]], "at timestep", effect[0] + 1 
for t in range(100,80,-1):
	causal_factor = (t,0,1)
	print "cause x is", headers[causal_factor[2]], "at timestep", causal_factor[0]
	interventions(lstm, output_layer, feature_index=causal_factor, output_index = effect, data=test_set)
	causal_strength(lstm, output_layer, feature_index=causal_factor, output_index = effect)

