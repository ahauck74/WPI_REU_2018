import torch, torch.nn as nn, torch.optim as optim
import time
from TensorCreation import *
from Models import *

###########################################################
#Training, Testing
###########################################################
class Training():
    def __init__(self, train, dev, test, config, tensor_config, train_config):
        self.train_data = train[0]
        self.train_labels = train[1]
        self.dev_data = dev[0]
        self.dev_labels = dev[1]
        self.test_data = test[0]
        self.test_labels = test[1]
        self.criterion = nn.BCELoss()
        self.batch_size = train_config["batch_size"]
        self.epochs = train_config["epochs"]
        self.learning_rate = train_config["learning_rate"]
        self.momentum = train_config["momentum"]
        self.use_shuffling = train_config["use_shuffling"]
        self.n_hidden = train_config["n_hidden"]
        self.model = train_config['model'].lower()
        self.use_masking = tensor_config["use_masking"]
        self.n_features = config["n_features"]
        self.n_classes = config["n_classes"]
    


    def choose_model(self, mean):
        if self.use_masking:
            features = int(3*self.n_features)
        else: features = self.n_features
        if self.model == 'rnn':
            self.rnn = MyRNN(features, self.n_hidden)
        elif self.model == 'lstm':
            self.rnn = MyLSTM(features, self.n_hidden)
        elif self.model == 'gru':
            self.rnn = MyGRU(features, self.n_hidden)
        elif self.model == 'grumanual':
            self.rnn = ManualGRU(features, self.n_hidden)
        elif self.model == 'grud':
            self.rnn = GRU_D(features, self.n_hidden)
            self.rnn.set_mean(mean)
        else:
            print('Model not valid.')
        self.optimizer = optim.SGD(self.rnn.parameters(), lr=self.learning_rate, momentum=self.momentum)

    
    
    '''
    The kernel randomly dies in the result method. There isn't any consistency to where in the method it dies.
    '''
    #Checks which value of the tensor is larger and creates a guess tensor accordingly: either 0,1 or 1,0
    def result(self, output):
        guess_tensor = torch.zeros(1, self.n_classes)
        if output[0][0].item() >= output[0][1].item():
            guess_tensor[0][0] = 1
        else:
            guess_tensor[0][1] = 1
        return(guess_tensor)



    #Input the entire data and label tensors
    #Call this method in a for loop as
    '''for iter in range(n_iters):
            if iter%batchSize == 0 and iter > 0:
                dataBatch, labelsBatch = makeBatch(trainDataTensor, trainLabelsTensor, iter, batchSize)'''
    def make_batch(self, data_tensor, labels_tensor, iter, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        data_batch = torch.zeros(data_tensor.shape[0], batch_size, data_tensor.shape[2], dtype=torch.float32)#Sizes of the tensors
        labels_batch = torch.zeros(batch_size, labels_tensor.shape[1])#####that get outputted
        data_batch[:,:batch_size,:] = data_tensor[:,iter-batch_size:iter,:]
        labels_batch[:batch_size,:] = labels_tensor[iter-batch_size:iter,:]
        return(data_batch, labels_batch)
    
    #For the data
    def shuffle_data(self, permutation, tensor):
        copy = torch.zeros(tensor.shape, dtype=torch.float32)
        for i in range(len(permutation)):
            j = permutation[i]
            copy[:,j] = tensor[:,i]
        return copy

    #For the labels
    def shuffle_labels(self, permutation, tensor):
        copy = torch.zeros(tensor.shape)
        for i in range(len(permutation)):
            j = permutation[i]
            copy[j] = tensor[i]
        return copy

    def shuffle_main(self, data, labels):
        size = data.shape[1]
        permutation = np.random.permutation(size)
        return self.shuffle_data(permutation, data), self.shuffle_labels(permutation, labels)

    def train_batch(self, data_tensor, class_tensor, hidden):
        self.rnn.zero_grad()
        output, hidden = self.rnn(data_tensor, hidden)
        loss = self.criterion(output, class_tensor)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return output, loss, hidden

    #Similar to train, but the doesn't update the parameters with gradient descent
    def dev_check(self, data_tensor, class_tensor, hidden):
        output, hidden = self.rnn(data_tensor, hidden)
        loss = self.criterion(output, class_tensor)

        return output, loss, hidden

    def timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    #Determines the accuracy
    def test_results(self):
        tot = 0
        correct = 0
        test_hidden = torch.zeros(1, self.test_data.shape[1], self.n_hidden, dtype=torch.float32)
        output, test_hidden = self.rnn(self.test_data.float(), test_hidden)
        for iter in range(self.test_data.shape[1]):
            if (output[iter][0] >= 0.5) == (self.test_labels[iter][0] >= 0.5):
                correct += 1
            tot += 1
        accuracy = correct/tot*100
        return(accuracy)
    
    
    def summarize_results(self, lossArray, devLossArray):
        #Creates the x axes for training and dev loss functions
        accuracy = self.test_results()
        print('accuracy', accuracy)
        lossTime = [x for x in range(len(lossArray))]
        devLossTime = [x for x in range(len(devLossArray))] 
        x1, y1 = np.asarray(lossTime), np.asarray(lossArray)
        x2, y2 = np.asarray(devLossTime), np.asarray(devLossArray)

        if(self.use_masking and self.impute_type == 'forward'):
            self.impute_type = 'simple'

        fig = plt.figure()

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        plt.scatter(x1, y1, c="g", alpha=0.5,label="Training")
        plt.scatter(x2, y2, c="b", alpha=0.5,label="Development")
        plt.legend(loc='best')

        ax.set_ylim([0,0.8])
        '''
        print('Version: ', version)
        print(model, self.impute_type, ', Masking: ', self.use_masking)
        print('Learning Rate: ', self.learning_rate, '| Momentum: ', self.momentum)
        print('Epochs:          ', self.epochs, ' | Batch Size: ', self.batch_size)
        print('Sequence Length: ', self.sequence_length, ' | Hidden Size: ', self.n_hidden)
        print('Shuffling:      ', self.use_shuffling, '| Noise: ', self.noise_flag)
        print('Missing:        ', self.missing_flag, '| Correlation: ', self.target_corr)
        self.printMode()
        print('Accuracy: ', accuracy, '%')
        '''

        plt.show()
        
        return(accuracy)

    def train_model(self):
        n_iters = self.train_data.shape[1]
        lossArray = []
        devLossArray = []
        start = time.time()
        for epoch in range(self.epochs):
            #print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            #print('epoch: ', epoch+1)
            #print(self.timeSince(start))
            if self.use_shuffling:
                self.train_data, self.train_labels = (self.shuffle_main(self.train_data, self.train_labels))
            hidden = torch.zeros(1, self.batch_size, self.n_hidden, dtype=torch.float32)#Initial hidden layer
            dev_hidden = torch.zeros(1, self.dev_data.shape[1], self.n_hidden, dtype=torch.float32)
            
            running_loss = 0
            losses = 0
            for iter in range(n_iters+1):
                if iter%self.batch_size == 0 and iter > 0:
                    data_batch, labels_batch = self.make_batch(self.train_data, self.train_labels, iter)
                    output, loss, hidden = self.train_batch(data_batch, labels_batch, hidden)
                    running_loss += loss.item()
                    losses += 1
                    
            avg_loss = running_loss/losses
            #print('loss: ', avg_loss)
            #print((epoch/self.epochs + iter/n_iters/self.epochs)*100, '%')
            lossArray.append(avg_loss)

            dev_output, dev_loss, dev_hidden = self.dev_check(self.dev_data.float(), self.dev_labels, dev_hidden)
            devLossArray.append(dev_loss.item())


        print('Training Complete')
        accuracy = self.summarize_results(lossArray, devLossArray)
        return(accuracy)
