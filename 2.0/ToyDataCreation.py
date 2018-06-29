import numpy as np
from abc import ABCMeta
import math, random
import matplotlib.pyplot as plt

#Algorithm

def pearson(series_1, series_2, sum1=None, sum2=None, den1=None, den2=None, num=None):
    
    size = len(series_1)
    
    '''First time through sum1, ... num are type None, so the except, which uses the whole series so far, must be used. 
       After the first run through, only the last value in the series(es) are needed to update the values so it isn't 
       necessary to loop through the entire series every time.'''
    try: 
        
        sum1 += series_1[-1]
        sum2 += series_2[-1]
        #print('sum')
        mean1 = sum1/size
        mean2 = sum2/size
        #print('mean')
        val1 = series_1[-1] - mean1
        val2 = series_2[-1] - mean2
        #print('val')
        den1 += val1*val1
        den2 += val2*val2
        num += val1*val2
        #print('fast')
        return num/(math.sqrt(den1)*math.sqrt(den2)), sum1, sum2, den1, den2, num
    
    except:
        sum1 = sum(series_1)
        sum2 = sum(series_2)
        mean1 = sum1/size
        mean2 = sum2/size

        num = 0
        den1 = 0
        den2 = 0
        for i in range(size):
            val1 = series_1[i] - mean1
            val2 = series_2[i] - mean2
            num += val1*val2

            den1 += val1*val1
            den2 += val2*val2
        #print('slow')
        if den1==0 or den2==0:
            return 0
        else:
            return num/(math.sqrt(den1)*math.sqrt(den2)), sum1, sum2, den1, den2, num
    
def abs_pearson(series_1, series_2, sum1=0, sum2=0, den1=0, den2=0, num=0):
    if(den1==0):
        return pearson(series_1,series_2)
    else:
        return pearson(series_1,series_2, sum1, sum2, den1, den2, num)

class dataset():
    """This class supports some basic methods useful across many datasets.
    
    This is mostly important for my personal experimental framework, but will be useful
    for this summer. One example of a useful method across datasets will be splitting data
    into training, development, and testing subsets. Sometimes, though, you may not want this,
    so it will be important not to call that function.
    """
    __metaclass__ = ABCMeta                                                                                                                       
                                                                                                                                                  
    def __init__(self, config):
        np.random.seed(3) # ensure repeatable results
        self.sequence_length = config["sequence_length"] # this controls the padding length for non-synthetic data
        self.n_classes = config["n_classes"] 
        self.n_features = config["n_features"]
        self.target_corr = config["target_corr"]
       
        

        # Set the train/dev/test split points
        self.split_proportions = [0.8, 0.1, 0.1] # % train, % dev, % test
        
    def get_data_summary(self):
        """Print a summary of the data generated"""
        print("--------------------------")
        print("Summarizing data generated")
        print("--------------------------")
        print()
        print("--------")
        print("Training")
        print("--------")
        print("   Sequence_length:     ", self.train[0].shape[0])
        print("   Number of instances: ", self.train[0].shape[1])
        print("   Number of variables: ", self.train[0].shape[2])
        print("   Class balance:       ", np.sum(self.train[1][:, 0])/len(self.train[1]))
        print("-----------")
        print("Development")
        print("-----------")
        print("   Sequence_length:     ", self.dev[0].shape[0])
        print("   Number of instances: ", self.dev[0].shape[1])
        print("   Number of variables: ", self.dev[0].shape[2])
        print("   Class balance:       ", np.sum(self.dev[1][:, 0])/len(self.dev[1]))
        print("-------")
        print("Testing")
        print("-------")
        print("   Sequence_length:     ", self.test[0].shape[0])
        print("   Number of instances: ", self.test[0].shape[1])
        print("   Number of variables: ", self.test[0].shape[2])
        print("   Class balance:       ", np.sum(self.test[1][:, 0])/len(self.test[1]))
        
    def split(self):
        """Split data into train/dev/test sets."""
        self.X_train, self.y_train, self.signal_train =  self.data[:, :self.split_points[0], :], self.labels[:self.split_points[0]], self.signal_locations[:self.split_points[0]]
        self.X_dev, self.y_dev, self.signal_dev = self.data[:, self.split_points[0]:self.split_points[1], :], self.labels[self.split_points[0]:self.split_points[1]], self.signal_locations[self.split_points[0]:self.split_points[1]]
        self.X_test, self.y_test, self.signal_test = self.data[:, self.split_points[1]:, :], self.labels[self.split_points[1]:], self.signal_locations[self.split_points[1]:]
        return (self.X_train, self.y_train, self.signal_train), (self.X_dev, self.y_dev, self.signal_dev), (self.X_test, self.y_test, self.signal_test)
        
class ToyData(dataset):
    """This class creates simple synthetic univariate time series data.
    
    This is the simplest version of a time series with which we can test deep learning architectures,
    thus ensuring that they are working properly.
    
    Positively-labeled time series have value 0 everywhere with value 1 at some timestep
    while negatively-labeled time series have value 0 at each timestep.
    
    For example, a time series with 10 timesteps and a signal at timestep 4 will look like:
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    while a negative time series with 10 timesteps will look like:
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      
    This class can be called assuming you have also defined dictionary `config' which contains
    values "sequence_length", "n_features", and "n_classes". This is so that for each experiment
    we can save one config file summarizing the experimental settings.
      
    Example
    -------
    >>> config = {
    ...             "sequence_length" : 10
    ...             "n_features"      : 1
    ...             "n_classes"       : 2
    ...         }
    >>> d = ToyData(config)
    >>> print(d.train[0]) # training time series
    >>> print(d.train[1]) # training labels
    
    """
    
    
    
    ###########################################################################################################
    
    ###########################################################################################################
    def __init__(self, config, n_features=1):
        super(ToyData, self).__init__(config=config)
        self.name = "ToyData"                       # Used for naming log files
        self.n_examples = 1500           #Set to 0 for no correlation between missingness and label,
                                                #Set to 1 for medium and 2 for high correaltion
        self.location_sampling_mode = "left_skew"     # When generating the time series, how should signals be distributed?
                                                        # Options include: uniform, normal, left_skew, right_skew

        self.split_points = [int(self.n_examples * i) for i in self.split_proportions]
        self.split_points = np.cumsum(self.split_points)
        self.signal_locations = self.set_signal_locations(self.location_sampling_mode) # create signal locations
        self.generate_data() # generate data
        self.train, self.dev, self.test = self.split() # split data into train/dev/test
        self.missing_flag = config["missing_flag"]
        self.noise_flag = config["add_noise"]
        
        
        if self.missing_flag:
            self.train, self.test, self.dev = self.remove_values(self.train, self.test, self.dev)
            
        if self.noise_flag: # if we want to add noise, that flag will be tripped here
            self.train = self.add_noise(self.train)
            self.dev = self.add_noise(self.dev) 
            self.test = self.add_noise(self.dev)
            
    
    ###########################################################################################################
    
    ###########################################################################################################
        
    def printMode(self):
        print("Data Mode:    ", self.location_sampling_mode)
        
    def save_training_data(self):
        data = self.train[0]
        labels = self.train[1]
        np.save('exported_synthetic_data', data)
        np.save('exported_synthetic_labels', labels)
        
        
    def add_noise(self, data):
        """Add noise to a time series dataset.
        
        Parameters
        ----------
        data : tuple
            This argument contains a dataset tuple (X, y) where X contains the time series
            and y contains the labels. X and y are both numpy arrays.
            
        Returns
        -------
        tuple
            New version of data, stored in the same way. However, noise has been added which should obfuscate
            the signal depending on how much noise is added (controlled by {mean, std})
        """
        X = data[0]
        y = data[1]
        mean = 0.0
        std = 0.1
        noise = np.random.normal(mean, std, int(X.shape[0]*X.shape[1]*X.shape[2])).reshape(X.shape)
        X = X + noise
        return (X, y)
    
    
    '''Adds 1-2 missing values to half of the time series. If the time series has a positive signal and
       the correlation value isn't zero, then more values are made missing for that time series with the 
       amount dependent on the correlation value. This way more missing values ==> positive result.'''
    
    
    #Algorithm

    

    
    
    def remove_percentage(self, data):
        var = 0.1
        label = np.zeros(data[0].shape[1])
        missingness = np.zeros(data[0].shape[1])
        for i in range(5):
            if data[1][i,0] == 1:
                missingness[i] = max(0,min(self.sequence_length - 1, int(np.random.normal(7, var))))
            else:
                missingness[i] = min(self.sequence_length - 1, max(0,int(np.random.normal(3, var))))
                
        for i in range(5,data[0].shape[1]):
            try:
                corr, sum1, sum2, den1, den2, num = abs_pearson(data[1][:i,0],missingness[:i], sum1, sum2, den1, den2, num)
            except:
                corr, sum1, sum2, den1, den2, num = abs_pearson(data[1][:i,0],missingness[:i])
                
            if corr > self.target_corr: var += 0.1
            else: var = max(0, var - 0.1)

            if data[1][i,0] == 1:
                missingness[i] = max(0, min(self.sequence_length - 1, int(np.random.normal(7, var))))
            else:
                missingness[i] = min(self.sequence_length - 1, max(0,int(np.random.normal(3, var))))
                
        return missingness
      
    
    def missing_graph(self, data, missArray):
        positive = []
        negative = []
        for i in range(data[0].shape[1]):
            if data[1][i,0] == 1: positive.append(missArray[i])
            else: negative.append(missArray[i])
        bins = [x for x in range(10)]
        plt.hist(positive, bins, alpha=0.5, label='positive')
        plt.hist(negative, bins, alpha=0.5, label='negative')
        plt.legend(loc='upper right')
        plt.xlabel = ('Number of Missing Values')
        plt.ylabel = ('Frequency')
        plt.show()
    
    
    def remove_values(self, train, test, dev): 
        data = [0]
        data[0] = np.append(train[0], test[0], axis = 1)
        data.append(np.append(train[1], test[1], axis = 0))
        data[0] = np.append(data[0], dev[0], axis = 1)
        data[1] = np.append(data[1], dev[1], axis = 0)
        
        
        missArray = self.remove_percentage(data)
        #self.missing_graph(data, missArray)
        
        for i in range(data[0].shape[1]):
            n_miss = missArray[i]
            while n_miss > 0:
                index = random.randint(0,self.sequence_length-1)
                if data[0][index,i,0].item() != 1 and not np.isnan(data[0][index,i,0].item()):
                    data[0][index,i,0] = np.nan
                    n_miss = n_miss - 1
                    
        new_train = [0]*2
        new_test = [0]*2
        new_dev = [0]*2
        
        big = int(0.8*self.n_examples) # for splitting array
        small = int(0.1*self.n_examples)
        
        array = np.split(data[0],[big,big+small], axis=1)
        new_train[0], new_dev[0], new_test[0] = array[0], array[1], array[2]
        array = np.split(data[1],[big,big+small], axis=0)
        new_train[1], new_dev[1], new_test[1] = array[0], array[1], array[2]

        return new_train, new_test, new_dev
    
                                                                           
    def set_signal_locations(self, mode="left_skew"):
        """Create a vector of timestep indices to impute signals into time series.
        
        By building this vector first, as opposed to randomly sampling for each instance,
        we are able to distribute the signal locations as we see fit, then compare which models
        are able to match this distribution. After sampling, all values are rounded to the nearest
        integer and constrained to be between 0 and the maximum sequence length.
        
        Half of the data are labeled "positive" (1) and include a signal, the other half
        are labeled "negative" (0). The task of a classifier will be to correctly predict
        the label of a time series it has never seen before. The labels will be added at
        understandable locations in each positive time series. By controlling these locations,
        we understand whether or not an Early Classifier can identify that correct halting-point,
        effectively matching the distribution of signal locations.
        
        Parameters
        ----------
        mode : str
            Indicates what sort of random sampling to implement.
              - "uniform"    - Sample from 0 to sequence_length from a uniform distribution
                             (i.e. count(signals_at_timestep_1) = count(signals_at_timestep_5))
              - "normal"     - Sample from a normal distribution centered at the halfway point
              - "left_skew"  - Sample from a normal distribution centered at 0,
                              then values below 0 are rounded up to 0.
              - "right_skew" - Sample from a normal distribution centered at the max sequence length,
                               then values above the max sequence length are rounded down to the max. length.
                               
        Returns
        -------
        np.array
            A list of signal locations, one for each time series. The first half will be used to place signals
            in the first half of the time series.  
        """
        if mode == "uniform":
            signal_locations = np.random.randint(self.sequence_length, size=int(self.n_examples))
        elif mode == "normal":
            signal_locations = np.round(np.random.normal(loc=int(self.sequence_length/2),
                                                         scale=2,
                                                         size=int(self.n_examples))).astype(np.int32)
        elif mode == "left_skew":
            signal_locations = np.round(np.random.normal(loc=0,
                                                         scale=int(self.sequence_length/2),
                                                         size=int(self.n_examples))).astype(np.int32)
        elif mode == "right_skew":
            signal_locations = np.round(np.random.normal(loc=self.sequence_length,
                                                         scale=int(self.sequence_length/2),
                                                         size=int(self.n_examples))).astype(np.int32)
        else:
            raise NotImplementedError
            
        signal_locations = [max(min(i, self.sequence_length-1), 0) for i in signal_locations]
        signal_locations = np.array([int(i) for i in signal_locations])
        
        return signal_locations
        
    def plot_signal_histogram(self):
        """Display a histogram of the locations of the imputed signals.
        
        Example
        -------
        >>> d = ToyData(config)
        >>> d.plot_signal_histogram()
        """
        fig, ax = plt.subplots()
        ax.hist(self.signal_locations[self.signal_locations != -1],
                bins=np.arange(0, self.sequence_length+1, 1),
                color='k')
        plt.title("Signals generated by a " + self.location_sampling_mode + "distribution")
        plt.ylabel("Signal count")
        plt.xlabel("Timestep")
        plt.tight_layout()
        plt.show()
        
    def generate_data(self):
        """Generate time series data with simple signals added throughout the timesteps."""
        
        X = np.zeros((self.sequence_length, self.n_examples, self.n_features)) # tensor of zeros of shape <sequence_length x number_of_examples x number_of_features>
        y = np.zeros((self.n_examples, 1)) # tensor of zeros that will contain integer labels, one for each example

        for i in range(int(self.n_examples/2)): # for the first half of the dataset, add signals according to extant signal_location array
            X[self.signal_locations[i], i, 0] = 1.0 # change 0 to 1
            y[i] = 1 # also list this example as class 1 (default is class 0)
            
        self.signal_locations[int(self.n_examples/2):] = -1 # For negative examples, store meaningless locations  
        X = X.transpose(1, 0, 2) # X must be transposed since np.random.shuffle shuffles along axis 0.
        zipped_data = list(zip(X, y, self.signal_locations)) # Align X, y, and locations
        np.random.shuffle(zipped_data) # shuffle all together so that dataset is not predictable
        X, y, self.signal_locations = zip(*zipped_data) # unzip after shuffle, each element is a tuple
        self.signal_locations = np.asarray(self.signal_locations) # cast to numpy array
        self.data = np.asarray(X).transpose(1, 0, 2).astype(np.float32) # Cast X as a numpy array of float32 vals, transpose back to proper shape
        self.labels = np.array(y).astype(np.int32) # Cast y as a numpy array of int's
        return(X,y)
