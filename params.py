

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
        self.data_format = 'mp3'

        # Model
        self.n_bins = 64
        self.n_frames = 96
        self.n_classes = 10

        # Training
        self.n_epochs = 5
        self.batch_size = 100
        self.val_split = .2
        self.data_root = '../data/largeBirds2_processed'
        self.n_max = 100
        self.weights = 'models/vggish_audioset_weights_without_fc2.h5'

        # model zoo
        self.model_zoo = 'models'
        self.name = 'BirdDetector_10classes'

        # computing device, can be 'cuda:<GPU index>' or 'cpu'
        self.device = 'cpu'
