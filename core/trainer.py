from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch

class Trainer():
    '''
    class to provide framework for training a network.
    Attributes:
        n_epochs (int): number of epochs of training
        model (nn.Module instance): model to be trained
        dataset (instance of DataSet subclass): to handle data
        train_loader (DataLoader instance): loader handling training data
        test_loader (DataLoader instance): loader handling test data
        criterion (nn.Loss function): loss function for training
        optimizer (nn.optim instance): optimizer for model params
        batch_size (int): batch size
        save_params (string): path to where
    '''

    def __init__(self, params, model, dataset, criterion, optimizer):
        '''
        initializes training from a parameter class object
        Args:
            model (nn.Module subclass instance): model to be training
            dataset (DataSet subclass instance): providing data
            criterion (nn.Loss instance): loss function
            optimizer (nn.optim instance): optimizer for model parameters
            params (params class instance): providing all relevant parameters
        '''
        self.model = model
        # training
        self.n_epochs = params.n_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = params.batch_size
        self.device = torch.device(params.device)
        # data loading
        self.dataset = dataset
        self.val_split = params.val_split
        self._init_dataloaders()
        # state
        self.current_epoch = 0
        # stats
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []


    def _init_dataloaders(self):
        '''
        from self.dataset initializes self.train_loader and self.test_loader
        according to params
        '''
        # make list of indices
        indices = list(range(len(self.dataset)))
        np.random.shuffle(indices) # is necessary b/c inidices are ordered by labels
        split_idx = int(np.floor(self.val_split*len(self.dataset)))
        train_indices = indices[split_idx:]
        val_indices = indices[:split_idx]
        # make samplers
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(val_indices)
        # make dataloaders
        self.train_loader = DataLoader(self.dataset, self.batch_size, sampler=train_sampler)
        self.test_loader = DataLoader(self.dataset, self.batch_size, sampler=test_sampler)


    def _train(self):
        '''
        performs one training iteration through the training data plus backpropagation
        '''
        self.model.train()
        epoch_loss = 0.
        total_correct = 0
        for i, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach().cpu().numpy()
            pred = out.argmax(1)
            total_correct += (pred == target).sum().cpu().numpy()
        self.train_acc.append(total_correct / len(self.train_loader.sampler))
        self.train_loss.append(epoch_loss / len(self.train_loader.sampler))


    def _test(self):
        '''
        performs one iteration through the evaluation set
        '''
        self.model.eval()
        test_loss = 0.
        total_correct = 0
        for i, (data, target) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, target)
            test_loss += loss.detach().cpu().numpy()
            pred = out.argmax(1)
            total_correct += (pred == target).sum().cpu().numpy()
        self.test_acc.append(total_correct / len(self.test_loader.sampler))
        self.test_loss.append(test_loss / len(self.test_loader.sampler))


    def run_training(self):
        '''
        executes a training routine over self.n_epochs
        '''
        for e in range(self.n_epochs):
            print('\nEpoch:', e)
            self._train()
            #print('training loss:', train_loss)
            print('training accuracy:', self.train_acc[-1])
            self._test()
            #print('test loss:', test_loss)
            print('test accuracy:', self.test_acc[-1])
