import sys
import torch
from torch.utils.data import DataLoader 
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import pandas as pd
import numpy as np

# -------------------------------
# the AutoEncoder class
# -------------------------------
        
class AutoEncoder:

    """ AutoEncoder model designed for anomaly detection """

    def __init__(self, input_dim, hidden_size, batch_size, learning_rate = 0.01, num_epochs = 100, run_in_gpu = True,logpath=None):
        """ init function"""

        # AE model parameters
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = torch.device("cpu")
        if run_in_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoEncoderModule(self.input_dim,self.hidden_size,self.device)
        
        # setting the training parameters
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # to log the loss of the model
        self.writer = None
        if logpath:
            log = 'AE_input_'+str(self.input_dim)+'_hidden_'+str(self.hidden_size)+'_lr_'+str(self.lr)+'_batchSz_'+str(self.batch_size)+'_numEp_'+str(self.num_epochs)
            self.writer = SummaryWriter(logdir=logpath +'/'+ log)

        # anomaly score normalizing constants
        self.max_err = None
        self.min_err = None
        
        return

    def fit(self,X):
        """ training the AutoEncoder model """

        # fits the model
        X = torch.tensor(X,dtype=torch.float)
        train_loader = DataLoader(dataset=X, batch_size=self.batch_size, drop_last=True, shuffle = True, pin_memory=True)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)

        for epoch in range(self.num_epochs):
            for ts_batch in train_loader:
                output, _  = self.model(self.to_var(ts_batch))
                
                #print(ts_batch, output)
                #sys.exit()
                loss = torch.nn.MSELoss(reduction='sum')(output,self.to_var(ts_batch.float()))

                #print(epoch,loss.cpu())
                #print(loss)
                #sys.exit()
                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                # logging the loss
                if self.writer:
                    self.writer.add_scalar('train_loss', loss.cpu() / self.batch_size , epoch + 1)

        if self.writer:
            self.writer.close()

        # gets the anomaly scores for fitted model
        # this is done to obtain the max/min error value
        # such that the errors from the model can be normalized [0,~1]

        #print(X.shape[0])
        train_anomaly_loader = DataLoader(dataset=X, batch_size= X.shape[0])
        #all_errors = []
        #count = 0
        #self.max_err, self.min_err = -np.inf, np.inf
        with torch.no_grad():
            for idx, ts in enumerate(train_anomaly_loader):
                #print(ts.shape)
                #print(idx)
                #count += 1
                output, _ = self.model(self.to_var(ts))

                #print(output)
                #print(ts)

                #sys.exit()
                error = torch.nn.L1Loss(reduction='none')(output,self.to_var(ts.float()))
                error = torch.sum(error,1) #need
                #print(error.shape)

                error = error.cpu().data.numpy()
                #all_errors.append(error)
                #print(idx, error)
                #all_errors.append(error)
        #print(error.shape)
        #print(error)
        self.min_err = np.min(error)
        self.max_err = np.max(error)
                
        return self.min_err, self.max_err

    def predict(self,X):
        """ complete pass for obtaining the anomaly score """

        X = torch.tensor(X,dtype=torch.float)
        test_loader = DataLoader(dataset=X, batch_size= X.shape[0])
        
        with torch.no_grad():
            for idx, ts in enumerate(test_loader):     
                # does one forward pass
                output, _ = self.model(self.to_var(ts))
                error = torch.nn.L1Loss(reduction='none')(output,self.to_var(ts.float()))
                error = torch.sum(error,1)

                # gets the anomaly score, normalized
                anomaly_scores = (error - self.min_err)/(self.max_err - self.min_err)
                return anomaly_scores.cpu().data

    def save(self,savepath):
        """ saves the model state + max/min loss for the fitted data (used for normalized anomaly score) """
        # save model 
        name = 'AE_input_'+str(self.input_dim)+'_hidden_'+str(self.hidden_size)+'_lr_'+str(self.lr)+'_batchSz_'+str(self.batch_size)+'_numEp_'+str(self.num_epochs)
        torch.save(self.model.state_dict(),savepath + '/' + model_name)
        
        # save model min/max err when fitted
        savename = savepath + '/' + 'normalized_constants'
        np.savez_compressed(savename,min_err = self.min_err, max_err = self.max_err)
            
        return

    def load(self,loadpath,name):
        """ loading the model state + anomaly score values (for normalizing)"""

        # load model state
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(loadpath+'/'+name,'cpu'))
        elif self.device == 'gpu':
            self.model.load_state_dict(torch.load(loadpath+'/'+name))
            self.model.to(self.device)

        # evaluate the model
        self.model.eval()
        
        # load normalize constants
        filename = np.load(loadpath +'/normalized_constants')
        data = np.load(filename)
        self.min_err = data['min_err']
        self.max_err = data['max_err']
        
        return

    def to_var(self, t, **kwargs):
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def print_model(self):
        """ prints the model architecture """
        print(self.model)
        return

# -------------------------------
# the AutoEncoder NN Module
# -------------------------------

class AutoEncoderModule(torch.nn.Module):

    """ the pytorch Neural Network module of the AutoEncoder"""
    
    def __init__(self,input_dim, hidden_size, device):
        """ init function"""
        
        super(AutoEncoderModule,self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = device
        #if run_in_gpu:
        #    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # the model
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_dim))
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_dim]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[torch.nn.Linear(int(a), int(b), bias = True)] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = torch.nn.Sequential(*layers)
        #if run_in_gpu:
        #    self.to_device(self._encoder)

        layers = np.array([[torch.nn.Linear(int(a), int(b), bias = True)] for a, b in dec_setup.reshape(-1, 2)]).flatten()[1:]
        self._decoder = torch.nn.Sequential(*layers)
        #if run_in_gpu:
        #    self.to_device(self._decoder)
    
        return

    def forward(self, ts_batch):
        """ 
        forward pass
        returns reconstructed_sequence and the hidden_state
        """
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return reconstructed_sequence, enc

    def to_device(self,module):
        module.to(self.device)
        return
