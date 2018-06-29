import torch
from ToyDataCreation import *



###########################################################
#Tensor Configuration
###########################################################

class TensorCreation(ToyData):
    def __init__(self, config):
        super(TensorCreation, self).__init__(config=config)

    
    #turns the 1x1 label tensors into 1x2 binary label tensors
    def binary_class_tensor(self, tensor):
        class_tensor = torch.zeros(tensor.shape[0], 2)#8000, 2
        for i in range(tensor.shape[0]):
            if tensor[i].item() == 1:
                class_tensor[i][0] = 1
            else:
                class_tensor[i][1] = 1
        return(class_tensor)


    def set_tensor_config(self, tensor_config):
        self.use_masking = tensor_config["use_masking"] and self.missing_flag
        self.impute_type = tensor_config["impute_type"]

    def make_tensors(self, data, labels):
        data_tensor = torch.from_numpy(data)
        class_tensor = torch.from_numpy(labels)
        class_tensor = self.binary_class_tensor(class_tensor)
        return(data_tensor, class_tensor)
    
    
    #This version appends masking vectors as additional features
    def masking_vector(self, data_tensor):
        time_stamps = [x for x in range(10)] #This is temporary
        new_data_tensor = torch.zeros(data_tensor.shape[0], data_tensor.shape[1], 3*data_tensor.shape[2], dtype=torch.float32)
        for k in range(data_tensor.shape[2]):
            for j in range(data_tensor.shape[1]):
                for i in range(data_tensor.shape[0]):
                    new_data_tensor[i,j,3*k] = data_tensor[i,j,k]
                    if not np.isnan(data_tensor[i,j,k]):
                        new_data_tensor[i,j,(3*k)+1] = 1 #Masking 1 if not missing

                    else:
                        new_data_tensor[i,j,(3*k)+1] = 0 #Masking 0 if missing

                    if not np.isnan(data_tensor[i-1,j,k]):
                        new_data_tensor[i,j,(3*k)+2] = max(0, time_stamps[i] - time_stamps[i-1])

                    else:
                        new_data_tensor[i,j,(3*k)+2] = max(0, new_data_tensor[i-1,j,3*k+2] + time_stamps[i]-time_stamps[i-1])
                    
        return new_data_tensor
    
    def calc_mean(self, data_tensor):
        num, denom, self.mean = [0]*self.n_features, [0]*self.n_features, [0]*self.n_features #For calculating the mean

        for k in range(self.n_features):
            iter = k
            if self.use_masking: k = 3*k
            for j in range(data_tensor.shape[1]):
                for i in range(data_tensor.shape[0]):
                    if not np.isnan(data_tensor[i,j,k]): 
                        num[iter] += data_tensor[i,j,k].item()
                        denom[iter] += 1
            try:
                self.mean[iter] = num[iter]/denom[iter]            
            except:
                self.mean[iter] = 0
                    
        

    def imputation(self, data_tensor):
            
        
        for k in range(self.n_features):
            iter = k #Used for iterating through the list of means in the case k needs to be tripled
            if self.use_masking: k = 3*k
            for j in range(data_tensor.shape[1]):
                for i in range(self.sequence_length):
                    if np.isnan(data_tensor[i,j,k]):
                        if (self.impute_type == 'forward'):
                            if i == 0:
                                data_tensor[i,j,k] = 0 #zero imputation if no previous value
                            else:
                                data_tensor[i,j,k] = data_tensor[i-1,j,k] #otherwise forward imputation
                                
                        elif (self.impute_type == 'mean'):
                            data_tensor[i,j,k] = self.mean[iter]
                            
                        elif (self.impute_type == 'decay') and (i==0):
                            data_tensor[i,j,k] == self.mean[iter] 

            
        return data_tensor 
    
    def create_tensors(self):
        #Create the tensors from the data using the above methods       
        train_data_tensor, train_labels_tensor = self.make_tensors(self.train[0], self.train[1])
        dev_data_tensor, dev_labels_tensor = self.make_tensors(self.dev[0], self.dev[1])
        test_data_tensor, test_labels_tensor = self.make_tensors(self.test[0], self.test[1])

        if self.use_masking and self.missing_flag:
            train_data_tensor = self.masking_vector(train_data_tensor) #Only the training is used to calculate the mean
            test_data_tensor = self.masking_vector(test_data_tensor) 
            dev_data_tensor = self.masking_vector(dev_data_tensor)
            
        
        if self.missing_flag:
            self.calc_mean(train_data_tensor) #Only the training is used to calculate the mean
            train_data_tensor = self.imputation(train_data_tensor) 
            test_data_tensor = self.imputation(test_data_tensor) 
            dev_data_tensor = self.imputation(dev_data_tensor)
            
        
        self.train = train_data_tensor, train_labels_tensor
        self.dev = dev_data_tensor, dev_labels_tensor
        self.test = test_data_tensor, test_labels_tensor
        
        return(self.train, self.dev, self.test, self.mean)

