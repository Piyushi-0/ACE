import joblib
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


num_features = 1
num_hidden = 1
output_size = 1

#defining the model
lstm = nn.GRU(num_features, num_hidden).cuda()
output_layer = nn.Linear(num_hidden, output_size).cuda()

loss_function = nn.MSELoss()
#loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.SGD([
                {'params': lstm.parameters()},
                {'params': output_layer.parameters()}
            ], lr = 0.001)


counter = 0
loss_val = 0
while(1):
        coin_flip = np.random.binomial(1,0.5,1)

        T = np.random.randint(10,15)
        input_data = np.random.normal(0.0,0.2,T)
        if coin_flip == 1:
            input_data[:3] = 1.0 + np.random.normal(0.0,0.2,3)
            y_label = 1.0
        else:
            input_data[:3] = -1.0 + np.random.normal(0.0,0.2,3)
            y_label = 0.0
      
        optimizer.zero_grad()

        # initialize the hidden state.
	hidden = autograd.Variable(torch.zeros(1, 1, num_hidden).type(torch.cuda.FloatTensor))
	input_torchvar = autograd.Variable(torch.cuda.FloatTensor(input_data), requires_grad=True)
        output, hidden = lstm(input_torchvar.view(len(input_data), 1, -1), hidden)

	output_2 = output_layer(output[-1])
	#print input_data[:3], output_2, train_y[idx]
        
	loss = loss_function(F.sigmoid(output_2).view(1), autograd.Variable(torch.cuda.FloatTensor([y_label])))
	#print loss.item()
        loss_val += loss.item()
        loss.backward()
        
        optimizer.step()
        counter += 1
        if counter%1000 == 0:
            print loss_val/1000.0
            loss_val = 0
            torch.save(lstm, "lstm_trained_model")
            torch.save(output_layer, "output_layer_trained_model")	
        
