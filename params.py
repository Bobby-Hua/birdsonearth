

class Params:
    '''
    defines all parameters needed in training.py and model.py

    TRAINING
    n_epochs (int): number of training epochs
    batch_size (int):
    val_split (float): fraction of data kept for validation
    data_root (str): path to data
    n_max (int): max number of instances loaded for training
    weights (str): path to vggish weights .hdf5 file
    '''

    def __init__(self):

        # Data
        #TODO: add options for other formats
        self.data_format = 'wav'

        # Model
        self.n_bins = 64
        self.n_frames = 96
        self.n_classes = 3

        # Training
        self.n_epochs = 100
        self.batch_size = 512
        self.val_split = .2
        self.data_root = 'data/full_urbansounds_restructured'
        # if mel_spec_root directory exists it is used and preprocessing of data_root is skipped
        # otherwise mel specs are computed from data_root
        self.mel_spec_root = 'data/full_urbansounds_specs'
        self.n_max = None
        self.weights = 'models/vggish_audioset_weights_without_fc2.h5'

        # model zoo
        self.save_model = False
        self.model_zoo = 'models'
        self.name = 'birds'

        # computing device, can be 'cuda:<GPU index>' or 'cpu'
        self.device = 'cuda:1'
