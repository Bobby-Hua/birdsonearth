import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax
from torch.utils.data import DataLoader
import h5py as h5
import os
import params as p



class VGGish(nn.Module):

    def __init__(self, params):

        super(VGGish, self).__init__()

        self.n_bins = params.n_bins
        self.n_frames = params.n_frames
        self.out_dims = int(params.n_bins / 2**4 * params.n_frames / 2**4)
        self.n_classes = params.n_classes
        self.weights = params.weights
        self.model_zoo = params.model_zoo
        self.name = params.name

        # convolutional bottom part
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # fully connected top part
        self.classifier = nn.Sequential(
            nn.Linear(self.out_dims*512, 1028),
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, self.n_classes)
        )

    def forward(self, X):

        a = self.pool1(relu(self.conv1(X)))
        a = self.pool2(relu(self.conv2(a)))
        a = relu(self.conv3_1(a))
        a = relu(self.conv3_2(a))
        a = self.pool3(a)
        a = relu(self.conv4_1(a))
        a = relu(self.conv4_2(a))
        a = self.pool4(a)
        a = a.reshape((a.size(0), -1))
        a = self.classifier(a)
        a = softmax(a)
        return a

    def init_weights(self, file=None):
        '''
        laods pretrained weights from an .hdf5 file. File structure must match exactly.

        Args:
            file (string): path to .hdf5 file containing VGGish weights
        '''

        if file is not None:
            file = file
        else:
            file = self.weights

        # loading weights from file
        with h5.File(file, 'r') as f:

            conv1 = f['conv1']['conv1']
            kernels1 = torch.from_numpy(conv1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases1 = torch.from_numpy(conv1['bias:0'][()])
            conv2 = f['conv2']['conv2']
            kernels2 = torch.from_numpy(conv2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases2 = torch.from_numpy(conv2['bias:0'][()])
            conv3_1 = f['conv3']['conv3_1']['conv3']['conv3_1']
            kernels3_1 = torch.from_numpy(conv3_1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases3_1 = torch.from_numpy(conv3_1['bias:0'][()])
            conv3_2 = f['conv3']['conv3_2']['conv3']['conv3_2']
            kernels3_2 = torch.from_numpy(conv3_2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases3_2 = torch.from_numpy(conv3_2['bias:0'][()])
            conv4_1 = f['conv4']['conv4_1']['conv4']['conv4_1']
            kernels4_1 = torch.from_numpy(conv4_1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases4_1 = torch.from_numpy(conv4_1['bias:0'][()])
            conv4_2 = f['conv4']['conv4_2']['conv4']['conv4_2']
            kernels4_2 = torch.from_numpy(conv4_2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases4_2 = torch.from_numpy(conv4_2['bias:0'][()])

            # assigning weights to layers
            self.conv1.weight.data = kernels1
            self.conv1.bias.data = biases1
            self.conv2.weight.data = kernels2
            self.conv2.bias.data = biases2
            self.conv3_1.weight.data = kernels3_1
            self.conv3_1.bias.data = biases3_1
            self.conv3_2.weight.data = kernels3_2
            self.conv3_2.bias.data = biases3_2
            self.conv4_1.weight.data = kernels4_1
            self.conv4_1.bias.data = biases4_1
            self.conv4_2.weight.data = kernels4_2
            self.conv4_2.bias.data = biases4_2

    def freeze_bottom(self):
        '''
        freezes the convolutional bottom part of the model.
        '''
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

    def save_weights(self):
        torch.save(self.state_dict(), \
                    os.path.join(self.model_zoo, self.name+'.pt'))
        return


class VGGish_w_top(nn.Module):

    def __init__(self, params, top=False):

        super(VGGish, self).__init__()

        self.n_bins = params.n_bins
        self.n_frames = params.n_frames
        self.out_dims = int(params.n_bins / 2**4 * params.n_frames / 2**4)
        self.n_classes = params.n_classes
        self.weights = params.weights
        self.weights_to = params.weights_to
        self.name = params.name
        self.top = top

        # convolutional bottom part
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        if self.top:
            self.fc1 = nn.Linear(self.out_dims*512, 4096)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, 128)

    def forward(self, X):

        a = self.pool1(relu(self.conv1(X)))
        a = self.pool2(relu(self.conv2(a)))
        a = relu(self.conv3_1(a))
        a = relu(self.conv3_2(a))
        a = self.pool3(a)
        a = relu(self.conv4_1(a))
        a = relu(self.conv4_2(a))
        a = self.pool4(a)
        return a

    def init_weights(self, file=None):
        '''
        laods pretrained weights from an .hdf5 file. File structure must match exactly.

        Args:
            file (string): path to .hdf5 file containing VGGish weights
        '''

        if file is not None:
            file = file
        else:
            file = self.weights

        # loading weights from file
        with h5.File(file, 'r') as f:
            conv1 = f['conv1']['conv1']
            kernels1 = torch.from_numpy(conv1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases1 = torch.from_numpy(conv1['bias:0'][()])
            conv2 = f['conv2']['conv2']
            kernels2 = torch.from_numpy(conv2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases2 = torch.from_numpy(conv2['bias:0'][()])
            conv3_1 = f['conv3']['conv3_1']['conv3']['conv3_1']
            kernels3_1 = torch.from_numpy(conv3_1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases3_1 = torch.from_numpy(conv3_1['bias:0'][()])
            conv3_2 = f['conv3']['conv3_2']['conv3']['conv3_2']
            kernels3_2 = torch.from_numpy(conv3_2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases3_2 = torch.from_numpy(conv3_2['bias:0'][()])
            conv4_1 = f['conv4']['conv4_1']['conv4']['conv4_1']
            kernels4_1 = torch.from_numpy(conv4_1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases4_1 = torch.from_numpy(conv4_1['bias:0'][()])
            conv4_2 = f['conv4']['conv4_2']['conv4']['conv4_2']
            kernels4_2 = torch.from_numpy(conv4_2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases4_2 = torch.from_numpy(conv4_2['bias:0'][()])

            # assigning weights to layers
            self.conv1.weight.data = kernels1
            self.conv1.bias.data = biases1
            self.conv2.weight.data = kernels2
            self.conv2.bias.data = biases2
            self.conv3_1.weight.data = kernels3_1
            self.conv3_1.bias.data = biases3_1
            self.conv3_2.weight.data = kernels3_2
            self.conv3_2.bias.data = biases3_2
            self.conv4_1.weight.data = kernels4_1
            self.conv4_1.bias.data = biases4_1
            self.conv4_2.weight.data = kernels4_2
            self.conv4_2.bias.data = biases4_2

    def freeze_bottom(self):
        '''
        freezes the convolutional bottom part of the model.
        '''
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

    def save_weights(self, path=None):
        if path is None:
            path = os.path.join(self.weights_to, self.name+'.pt')
        torch.save(self.state_dict(), path)


if __name__ == '__main__':

    params  = p.Params()

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
        print('GPU available, working on device', device)
    else:
        device = torch.device('cpu')
        print('No GPU available, working on CPU.')

    print('loading model')
    net = VGGish(params)
    #print('net on gpu?:', net.is_cuda)
    net.init_weights('vggish_audioset_weights_without_fc2.h5')
    net.to(device)
    t = torch.randn((10, 10), device=device)
    print(t.device)
    t2 = torch.randn((10, 10))
    t2.to(device)
    print(t.device)
    for name, param in net.named_parameters():
        print(name, param.device)



    # ONNX export
    #net.cuda()
    # dummy_in = torch.randn(size=(10, 1, 64, 96)).cuda()
    # input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(12)]
    # output_names = ["output1"]
    # torch.onnx.export(net, dummy_in, "modelsVGGish_conv.onnx", verbose=True, \
    #                     input_names=input_names, output_names=output_names)
