import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from utils import vggish_input

class MelSpecDataset(Dataset):
    '''
    A subclass of pytorch's Dataset class to handle bird sounds data as processed
    by the xeno-canto crawler class.
    '''

    def __init__(self, params):
        '''
        initializes the Data set to a root directory data_root
        '''
        print('initialization')
        self.root = params.mel_spec_root
        self.max_size = params.n_max # max #instances / class used for training
        self.labels = []
        self.n_instances = []
        self.df = pd.DataFrame(data={'label':[], 'file':[], 'path':[]})
        self.load_data()

    def load_data(self):
        '''
        iterates through the path given by self.root that must have structure
        root/{classes}/{chunks} where chunks must be ndarrays as produced by
        vggish_input and respective labels arrays and paths in self.df dataframe.
        '''
        label_list = os.listdir(self.root)
        label_list.sort()
        try:
            label_list.remove('.DS_Store')
        except:
            None
        for label in label_list:
            self.labels.append(label)
            paths = [os.path.join(self.root, label, file) \
                    for file in os.listdir(os.path.join(self.root, label))]
            if self.max_size is not None:
                paths = paths[:self.max_size]
            self.n_instances.append(len(paths))
            print('loading {} spectrograms for class {}'.format(self.n_instances[-1], label))
            arrays = [np.load(path) for path in paths]
            labels_for_files = [label]*len(arrays)
            label_df = pd.DataFrame(data={'label':labels_for_files, 'file':arrays, \
                                    'path':paths})
            self.df = self.df.append(label_df, ignore_index=True)
        print('done data loading')

    def __getitem__(self, idx):
        '''
        returns chunk at idx as defined in self.df dataframe plus its label as
        a tensor (1xN_framesxN_bins) and int resp.
        '''
        label = self.df.at[idx, 'label']
        array = self.df.at[idx, 'file']
        #array = np.load(os.path.join(self.root, label, file))
        tensor = torch.from_numpy(array)
        return tensor.unsqueeze(0).float(), self.labels.index(label)

    def __len__(self):
        return len(self.df.index)


def make_loaders(dataset, batch_size, val_split):
    '''
    takes a dataset and sets up train and test dataloaders for it

    Args:
        dataset (object of Dataset subclass):
        batch_size (int):
        val_split (float): fraction of data for kept for validation
        shuffle_data (bool): if true indices are shuffled befor dividing them

    returns:
        respective train and test loaders (objects of the DataLoader class)
    '''
    # make list of indices
    indices = list(range(len(dataset)))
    np.random.shuffle(indices) # is necessary b/c inidices are ordered by labels
    split_idx = int(np.floor(val_split*len(dataset)))
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    # make samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)
    # make dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

def process_xeno_canto_downloads(root, save_to):
    '''
    iterates through data under root that must have the structure
    root/{classes}/{instances}. Under save_to the same subdirectories save_to/{classes}
    are generated and for every instance vggish compatible inputs are generated according
    to the parameters specified in vggish_params.py

    Args:
        root (string): path to data root
        saveto (string): path to root where processed data is stored
    '''
    label_list = os.listdir(root)
    try:
        label_list.remove('.DS_Store')
    except:
        None
    for label in label_list:
        if not os.path.isdir(os.path.join(root, label)): continue
        print('prcossing data for label {}'.format(label))
        os.makedirs(os.path.join(save_to, label))
        for file in os.listdir(os.path.join(root, label)):
            print('processing:', os.path.join(root, label, file))
            data = vggish_input.wavfile_to_examples(os.path.join(root, label, file))
            for i in range(data.shape[0]):
                np.save(os.path.join(save_to, label, file[:-4]+str(i)+'.npy'), data[i])

if __name__ == '__main__':

    #process_xeno_canto_downloads(root='../data/largeBirds_butSmall', save_to='../data/largeBirds_butSmall_processed')
    #process_xeno_canto_downloads(root='data', save_to='data/processed')

    dataset = BirdSoundsDataset(data_root='../data/largeBirds_butSmall_processed')
    train_L, test_L = make_loaders(dataset, batch_size=64, val_split=.2)
    batch, targets = next(iter(train_L))
    print(batch.size())
    print(targets.size())
