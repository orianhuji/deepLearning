########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader_a as ld
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


batch_size = 32
output_size = 2
hidden_size = 8        # to experiment with

run_recurrent = False    # else run Token-wise MLP
use_RNN = False          # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 300

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        # self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2hidden = MatMul(input_size, hidden_size)
        self.prev_hidden2hidden = MatMul(hidden_size, hidden_size)
        self.hidden2out = MatMul(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        hidden_state = self.sigmoid(self.prev_hidden2hidden(hidden_state) + self.in2hidden(x))
        output = self.sigmoid(self.hidden2out(hidden_state))
        return output.squeeze(), hidden_state

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).cuda()

# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.update_gate = MatMul(input_size, hidden_size)
        self.reset_gate = MatMul(input_size, hidden_size)
        self.out_gate = MatMul(hidden_size, hidden_size)
        self.hidden2out = MatMul(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        update = torch.sigmoid(self.update_gate(x))
        reset = torch.sigmoid(self.reset_gate(x))
        combined_reset = reset * hidden_state
        hidden_state = update * hidden_state + (1 - update) * torch.tanh(self.out_gate(combined_reset))
        output = self.hidden2out(hidden_state)
        return output.squeeze(), hidden_state

    def init_hidden(self, bs):
        return self.hidden_size


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.sigmoid = torch.nn.Sigmoid()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size,hidden_size)
        self.layer2 = MatMul(hidden_size,hidden_size)
        self.layer3 = MatMul(hidden_size,output_size)
        # additional layer(s)
        

    def name(self):
        return "MLP"

    def forward(self, x):

        # Token-wise MLP network implementation
        
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        # rest

        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, atten_size=5):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)
        self.atten_size = atten_size
        
        # Token-wise MLP + Restricted Attention network implementation
        self.layer1 = MatMul(input_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.layer2 = MatMul(hidden_size, output_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(batch_size, num_words, input_size))

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Add positional encoding
        x = x + self.positional_encoding[:x.shape[0],:,:]

        # Token-wise MLP + Restricted Attention network implementation
        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends
        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))

        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :]

        # Applying attention layer
        queries = self.W_q(x).unsqueeze(2)
        keys = self.W_k(x_nei)
        values = self.W_v(x_nei)

        # Compute attention scores
        atten_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.sqrt_hidden_size
        atten_weights = self.softmax(atten_scores)

        # Apply attention weights
        atten_output = torch.matmul(atten_weights, values).squeeze(2)

        # Pass through final linear layer
        output = self.layer2(atten_output)

        return output, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    for i in range(len(rev_text)):
        print(f'for word: {rev_text[i]}, prediction was: {"1" if bool(sbs1[i] > sbs2[i]) else "0"}, true label is: {"1" if bool(lbl1 > lbl2) else "0"}')

# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if atten_size > 0:
        model = ExRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)

print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth"))

model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0

# training steps in which a test step is executed every test_interval

model.train()

train_losses = []
test_losses = []
accuracies = []

for epoch in range(num_epochs):

    itr = 0 # iteration counter within each epoch
    epoch_train_loss = 0
    epoch_test_loss = 0

    for labels, reviews, reviews_text in train_dataset:   # getting training batches

        # test if the model can predict allways False
        # labels = torch.arange(0, 2, dtype=torch.float32).repeat(1024,1).cuda()

        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch 
        else:
            test_iter = False

        # Recurrent nets (RNN/GRU)

        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE

        else:  

        # Token-wise networks (MLP / MLP + Atten.) 
        
            sub_score = []
            if atten_size > 0:  
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:               
                # MLP
                sub_score = model(reviews)

            output = torch.mean(sub_score, 1)
            
        # cross-entropy loss

        loss = criterion(output, labels)

        # optimize in training iterations

        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            epoch_test_loss += test_loss
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            epoch_train_loss += train_loss

        if test_iter:
            accuracy = accuracy_score(torch.argmax(output, dim=1).cpu(), torch.argmax(labels, dim=1).cpu())
            accuracies.append(accuracy)
            print(f"Accuracy: {accuracy}")
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}"
            )

            if not run_recurrent:
                nump_subs = sub_score.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                # print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

            # saving the model
            torch.save(model, model.name() + ".pth")
        
    train_losses.append(epoch_train_loss / len(train_dataset))
    test_losses.append(epoch_test_loss / (len(train_dataset) // test_interval))


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Losses')

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracies)), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.tight_layout()
plt.show()