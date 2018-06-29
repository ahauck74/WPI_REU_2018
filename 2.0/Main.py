from Models import *
from TensorCreation import *
from ToyDataCreation import *
from TrainerAndResults import *
import numpy as np


'''
This is where most of the parameters get changed. There are a few additional parameters that get changed 
in the Toy Data class such as correlation.
'''

seq = 10 #Length of the time series
         #This currently needs to stay at 10 is there are missing values. The methods which add missing values
         #are currently made specifically for a sequence length of 10
features = 1 #Number of variables before their masking vectors get added
classes = 2 #Number of possible classifications

batch_size = 256
epochs = 1500
lr = 0.001 #Learning rate
momentum = 0.9
model = 'gru' #rnn, gru, or lstm
use_masking = True #Use a masking vector?
impute_type = 'forward' #mean or forward, else it becomes zero
n_hidden = 8 #Number of hidden layers
use_shuffling = True #Shuffle before each epoch?
target_corr = 0.8
missing_flag = True
add_noise = True


config = {
    
    "sequence_length" : seq, # sequence length
    "n_features"      : features, # how many variables do we want
    "n_classes"       : classes, # how many possible classes do we want
    
    "target_corr"     : target_corr, #target correlation in range [0,1]
    "missing_flag"    : missing_flag,
    "add_noise"       : add_noise
}

tensor_config = {
    
    "use_masking"     : use_masking,
    "impute_type"     : impute_type,
    
    
    
}

train_config = {
    "batch_size"      : batch_size,
    "epochs"          : epochs,
    "learning_rate"   : lr,
    "momentum"        : momentum,
    "use_shuffling"   : use_shuffling,
    "n_hidden"        : n_hidden,
    "model"           : model
}



def run(d, config, tensor_config, train_config, model='rnn', impute_type='zero', use_masking=False, base=None, target_corr = 0.8):
    tensor_config["use_masking"] = use_masking
    tensor_config["impute_type"] = impute_type
    train_config["model"] = model

    d.set_tensor_config(tensor_config)
    train, dev, test, mean = d.create_tensors()
    trainer = Training(train, dev, test, config, tensor_config, train_config)
    trainer.choose_model(mean)
    accuracy = trainer.train_model()
    
    return(accuracy)

def create_data(config):
    config['target_corr'] = target_corr
    d = TensorCreation(config)
    return(d)
    
    
def main(model, config, tensor_config, train_config):
    accuracy_corr_00 = {}
    accuracy_corr_20 = {}
    accuracy_corr_50 = {}
    accuracy_corr_80 = {}
    
    accuracy_corr_00['gru_mean']= []
    accuracy_corr_00['gru_forward']= []
    accuracy_corr_00['gru_simple']= []
    accuracy_corr_00['gru_d']= []
    
    accuracy_corr_20['gru_mean']= []
    accuracy_corr_20['gru_forward']= []
    accuracy_corr_20['gru_simple']= []
    accuracy_corr_20['gru_d']= []
    
    accuracy_corr_50['gru_mean']= []
    accuracy_corr_50['gru_forward']= []
    accuracy_corr_50['gru_simple']= []
    accuracy_corr_50['gru_d']= []
    
    accuracy_corr_80['gru_mean']= []
    accuracy_corr_80['gru_forward']= []
    accuracy_corr_80['gru_simple']= []
    accuracy_corr_80['gru_d']= []
    
    
    models = ['gru', 'grud']
    impute_types = ['mean', 'forward', 'decay']
    use_masking = [False, True]
    base = ['gru_mean', 'gru_forward', 'gru_simple', 'gru_d']
    target_corr_list = [0, 0.2, 0.5, 0.8]
    
    d = create_data(config)
    for corr in target_corr_list:
        print(corr)
        corr = 0.8
        for i in range(10):
            print(i)
            mean = run(d, config, tensor_config, train_config, models[0], impute_types[0], use_masking[0], base[0], corr)#mean
            forward = run(d, config, tensor_config, train_config, models[0], impute_types[1], use_masking[0], base[1], corr)#forward
            simple = run(d, config, tensor_config, train_config, models[0], impute_types[1], use_masking[1], base[2], corr)#simple
            grud = run(d, config, tensor_config, train_config, models[1], impute_types[2], use_masking[1], base[3], corr)#decay
            
            if corr == 0:
                accuracy_corr_00['gru_mean'].append(mean)
                accuracy_corr_00['gru_forward'].append(forward)
                accuracy_corr_00['gru_simple'].append(simple)
                accuracy_corr_00['gru_d'].append(grud)
            elif corr == 0.2:
                accuracy_corr_20['gru_mean'].append(mean)
                accuracy_corr_20['gru_forward'].append(forward)
                accuracy_corr_20['gru_simple'].append(simple)
                accuracy_corr_20['gru_d'].append(grud)
            elif corr == 0.5:
                accuracy_corr_50['gru_mean'].append(mean)
                accuracy_corr_50['gru_forward'].append(forward)
                accuracy_corr_50['gru_simple'].append(simple)
                accuracy_corr_50['gru_d'].append(grud)
            elif corr == 0.8:
                accuracy_corr_80['gru_mean'].append(mean)
                accuracy_corr_80['gru_forward'].append(forward)
                accuracy_corr_80['gru_simple'].append(simple)
                accuracy_corr_80['gru_d'].append(grud)
    
    for model in base:
        print('corr 0', model, ' accuracy: ', np.mean(accuracy_corr_00[model]), 'stdev: ', np.std(accuracy_corr_00[model]))
        print('corr 20', model, ' accuracy: ', np.mean(accuracy_corr_20[model]), 'stdev: ', np.std(accuracy_corr_20[model]))
        print('corr 50', model, ' accuracy: ', np.mean(accuracy_corr_50[model]), 'stdev: ', np.std(accuracy_corr_50[model]))
        print('corr 80', model, ' accuracy: ', np.mean(accuracy_corr_80[model]), 'stdev: ', np.std(accuracy_corr_80[model]))
        
    return(accuracy_corr_00, accuracy_corr_20, accuracy_corr_50, accuracy_corr_80)
            
    

corr0, corr20, corr50, corr80 = main(model, config, tensor_config, train_config)
