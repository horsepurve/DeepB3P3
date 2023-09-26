"""
copyright@Chunwei Ma 2023
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import pickle
from scipy import sparse
from sklearn.metrics import accuracy_score, \
                            f1_score, \
                            matthews_corrcoef, \
                            roc_auc_score        

CNN_EMB = True # False # 

BATCH_SIZE = 16 # 16
NUM_CLASSES = 10
NUM_EPOCHS = 50
NUM_ROUTING_ITERATIONS = 1
CUDA = True # False # 
LR = 0.01

SAVE_MODEL = False
EVALUATE_ALL = False

BEST = [0,0]

RTdata_path = 'dia.pkl' # 'mod.pkl' # 'unmod.pkl' # 'SCX.pkl' # 
LOAD_DATA = True # False # 

from RTdata_emb import Dictionary, \
                       RTdata, \
                       Pearson, \
                       Spearman, \
                       Delta_t95, \
                       DATA_AUGMENTATION, \
                       Corpus
## dictionary = Dictionary(dict_path)

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    # print(transposed_input.contiguous().view(-1, transposed_input.size(-1)).shape)    
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)),dim=1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float() # Note float here!


class CapsuleLayer(nn.Module):
    def __init__(self, 
                 num_capsules, 
                 num_route_nodes, 
                 in_channels, 
                 out_channels, 
                 kernel_size=None, 
                 stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):

        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, 
                                                          num_route_nodes, 
                                                          in_channels, 
                                                          out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, 
                           out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            if True == CUDA:
                logits = Variable(torch.zeros(*priors.size())).cuda()
            if False == CUDA:
                logits = Variable(torch.zeros(*priors.size()))

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs

param_2D = {'data' : 'mnist',
            'dim' : 2,
            'conv1_kernel' : 9,
            'pri_caps_kernel' : 9,
            'stride' : 2,
            'digit_caps_nodes' : 32 * 6 * 6,
            'NUM_CLASSES' : 10}

param_1D = {'data' : 'mnist',
            'dim' : 1,
            'conv1_kernel' : (28, 9),
            'pri_caps_kernel' : (1, 9),
            'stride' : 1,
            'digit_caps_nodes' : 32 * 1 * 12,
            'NUM_CLASSES' : 1}

class CapsuleNet(nn.Module):
    def __init__(self,
                 param, dictionary,
                 conv1_kernel,
                 conv2_kernel,
                 max_length,
                 args=0):
        super(CapsuleNet, self).__init__()
        
        self.param = param

        if 0 == args:
            self.CLA = False
        else:
            self.CLA = args.CLASS # peptide classification
        
        EMB_SIZE = 0
        if True == CNN_EMB:
            # self.emb = nn.Embedding(len(dictionary), len(dictionary))
            # Note: if using embedding, EMB_SIZE can be any value, and we choose 20 here
            EMB_SIZE = 20
            self.emb = nn.Embedding(len(dictionary), EMB_SIZE) # we use 20 for all data
        else:
            # Note: if using one-hot encoding, EMB_SIZE must be the same as len(dictionary)
            EMB_SIZE = len(dictionary)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=EMB_SIZE, nhead=5, dim_feedforward=64, dropout=0.)
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=256, # 256
                               kernel_size=(EMB_SIZE, conv1_kernel), # param['conv1_kernel'], # (28, 9), # 9, 
                               stride=1)
        ''''''
        self.bn1 = nn.BatchNorm2d(256) # Note: do we need this or not?
        self.conv2 = nn.Conv2d(in_channels=256, 
                               out_channels=256, # 256
                               kernel_size=(1, conv1_kernel), # (28, 9), # 9, 
                               stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        '''
        print('>> use the 3rd Conv layer!')
        self.conv3 = nn.Conv2d(in_channels=128, 
                                out_channels=256, # 256
                                kernel_size=(1, conv1_kernel), # (28, 9), # 9, 
                                stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        '''

        self.primary_capsules = CapsuleLayer(num_capsules=8, # 8
                                             num_route_nodes=-1, 
                                             in_channels=256, # 256
                                             out_channels=32, # 32
                                             kernel_size=(1, conv2_kernel), # param['pri_caps_kernel'], # (1, 9), # 9, 
                                             stride=self.param['stride']) # 1) # 2)

        self.digit_capsules = CapsuleLayer(num_capsules=self.param['NUM_CLASSES'], # 1, #NUM_CLASSES, # DeepRT
                                           num_route_nodes=32 * 1 * (max_length - conv1_kernel*2 + 2 - conv2_kernel + 1), # param['digit_caps_nodes'], # 32 * 1 * 12, # 32 * 6 * 6, 
                                           in_channels=8, # 8
                                           out_channels=16) # max_length-conv1_kernel + 1) # 16

        # add dropout:
        # self.dropout = nn.Dropout(0.1) # not good!
        # self.linear = nn.Linear((max_length-conv1_kernel+1)*256,16) # try residue: not good!
        ''' residue is not very good!
        pad = 0
        kernel_h = pad*2+1 # len + pad*2 - (kernel_h - 1) = len
        self.conv_res = nn.Conv2d(in_channels=256, 
                                  out_channels=1, # 256
                                  kernel_size=(1, kernel_h), # (28, 9), # 9, 
                                  stride=1)
                                  #padding =(0,pad))
        '''
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        # print('>>dim: input', x.shape) # [batch, 1, 28, 28]
        # print('>>dim: y', y) # [batch, 10] ~ [batch, NUM_CLASSES]
        if True == CNN_EMB:
            x = self.emb(x) # [batch, len] -> [batch, len, dict]
            x = x.transpose(dim0=1, dim1=2) # -> [batch, dict, len]
            x = x[:,None,:,:] # -> [batch, 1, dict, len]

        # ^^^^^ pre-process x ^^^^^
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)      

        ''' try residue: not good! 
        residue = x.view(x.shape[0],-1)        
        residue = self.linear(residue).view(residue.shape[0],1,16)
        # another residue method
        residue = F.relu(self.conv_res(x), inplace=True)
        residue = residue.view(residue.shape[0],1,residue.shape[-1])
        '''

        # x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True) # improvement        
        # x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        # print('>>dim: conv1', x.shape) # [batch, 256, 20, 20]
        x = self.primary_capsules(x)
        # print('>>dim: primary_capsules', x.shape) # [batch, 1152, 8] = [batch, 6*6*32, 8]
        # print('>>dim: unsqueezeed', self.digit_capsules(x).shape) # [10, batch, 1, 1, 16] ~ [num_caps, batch, ...]
        if 2 == self.param['dim']:# or self.CLA:
            x = self.digit_capsules(x).squeeze().transpose(0, 1) # DeepRT
            # [10, batch, 1, 1, 16] -> squeeze: [10, batch, 16] -> transpose: [batch, 10, 16]
        elif 1 == self.param['dim']:
            x = self.digit_capsules(x).squeeze()[:, None, :]
            # [1, batch, 1, 1, 16] -> squeeze: [batch, 16]
        # print('>>dim: digit_capsules', x.shape) # [batch, 10, 16]

        # add dropout:
        # x = self.dropout(x)
        # x = self.linear(x)
        # x = F.sigmoid(x)
        
        # x = x + residue # try residue: not good!
        classes = (x ** 2).sum(dim=-1) ** 0.5
        # print('>>dim: classes', classes) # [batch, 10]
        if 2 == self.param['dim']:# or self.CLA:
            classes = F.softmax(classes) # DeepRT
        # print('>>dim: softmax', classes)

        if y is None: # Note: not do this during training. Here y is only used for reconstruction
            if 2 == self.param['dim']:
                # In all batches, get the most active capsule.
                # print('>>dim: reconstruction', classes) # [batch, 10]
                _, max_length_indices = classes.max(dim=1) 
                # give: [torch.FloatTensor of size batch] and [torch.FloatTensor of size batch]
                if True == CUDA:
                    y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
                if False == CUDA:
                    y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data)
                # generate a new y: [batch, 10] with each column having 1 in batch 0

        if 2 == self.param['dim']:
            # print('>>dim: x*y', x.shape, y.shape)
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
           # x: [batch, 10, 16], y: [batch, 10] -> [batch, 10, 1]
            return classes, reconstructions
        if 1 == self.param['dim']:
            return classes, x # Note here

class CapsuleLoss(nn.Module):
    def __init__(self, param, args):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        
        self.param = param

        self.CLA = args.CLASS # peptide classification
        self.lo = nn.BCELoss()        

    def forward(self, images, labels, classes, reconstructions):
        if 2 == self.param['dim']:
            # print('>>dim: labels', labels) # [batch, 10]
            # print('>>dim: classes', classes) # [batch, 10]
            left = F.relu(0.9 - classes, inplace=True) ** 2
            right = F.relu(classes - 0.1, inplace=True) ** 2

            margin_loss = labels * left + 0.5 * (1. - labels) * right
            margin_loss = margin_loss.sum()

            ## note: we do not consider reconstruction
            ## reconstruction_loss = self.reconstruction_loss(reconstructions, images)
            ## loss = (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
            loss = margin_loss / labels.shape[0]

            # print('>>dim: loss', loss) # it's a single value
            return loss
        elif 1 == self.param['dim']:
            # print('>>dim: labels', labels) # torch.cuda.FloatTensor of size batch x 1
            # print('>>dim: classes', classes) # [batch, 1]

            '''
            square = (labels - classes) ** 2
            square = square.sort(dim=0,descending=False)[0]
            cut = int(labels.shape[0]-1)
            loss = (square[:cut]).sum()/cut
            loss = loss ** 0.5 + square[cut] ** 0.5
            '''            
            if self.CLA:
                loss = self.lo(classes,labels)
            else:
                loss = ((labels - classes) ** 2).sum()/labels.shape[0] # MSE # Note: here it must be sum()
                loss = loss ** 0.5 # RMSE            

            # print('>>dim: loss', loss)
            return loss

def desparse(RTtt):
    if False == DATA_AUGMENTATION:
        X = np.zeros((RTtt.number_seq, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
        for i in range(RTtt.number_seq): # DATA_AUGMENTATION->*2
            # sparse to dense
            X[i,::] = RTtt.X[i].todense()
        RTtt.X = X
    else:
        print('>> note: usnig data_augmentation')
        X = np.zeros((RTtt.number_seq*2, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
        for i in range(RTtt.number_seq*2): # DATA_AUGMENTATION->*2
            # sparse to dense
            X[i,::] = RTtt.X[i].todense()
        RTtt.X = X

### if __name__ == "__main__":
def param_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', 
                        default='data/mod_train_2.txt', help="training samples")
    parser.add_argument('--test_path', 
                        default='data/mod_test_2.txt', help="testing samples")
    parser.add_argument('--result_path', 
                        default='result/mod/mod_test_2.pred.txt', help="write obse, pred")
    parser.add_argument('--log_path', 
                        default='result/mod_test_2.log', help="training log")
    parser.add_argument('--save_prefix', 
                        default='epochs', help="where to store pt")
    parser.add_argument('--pretrain_path', 
                        default='', help="transfer learning: pretrained pt")
    parser.add_argument('--dict_path', 
                        default='', help="modification: load AA dict")
    parser.add_argument('--conv1_kernel', 
                        default=10, type=int, help="conv1 kernel size")
    parser.add_argument('--conv2_kernel', 
                        default=10, type=int, help="conv1 kernel size")
    parser.add_argument('--min_rt', 
                        default=0, type=float, help="min rt")
    parser.add_argument('--max_rt', 
                        default=1, type=float, help="max rt")
    parser.add_argument('--time_scale', 
                        default=1, type=float, help="conv1 kernel size")
    parser.add_argument('--max_length', 
                        default=50, type=int, help="max peptide length")
    # default params:
    parser.add_argument('--CNN_EMB', action='store_true', 
                        default=True, help="use embedding")
    parser.add_argument('--BATCH_SIZE', 
                        default=16, type=int, help="batch size")
    parser.add_argument('--NUM_EPOCHS', 
                        default=50, type=int, help="number of epochs")
    parser.add_argument('--NUM_ROUTING_ITERATIONS', 
                        default=1, type=int, help="routings")
    parser.add_argument('--CUDA', action='store_true', 
                        default=True, help="use CUDA")
    parser.add_argument('--LR', type=float, 
                        default=0.01, help='LR')
    parser.add_argument('--SAVE_MODEL', action='store_true', 
                        default=False, help="save model as pt")
    parser.add_argument('--EVALUATE_ALL', action='store_true', 
                        default=False, help="save: train/test pred/feat")    
    parser.add_argument('--CLASS', action='store_true', 
                        default=False, help="classification mode: CE loss")
    parser.add_argument('--regCLASS', action='store_true', 
                        default=False, help="use regression but evaluate as classification: MSE loss")                            

    # fill parameters: ========== ========== ========== ========== ==========
    args = parser.parse_args()
    train_path    = args.train_path # 'data/mod_train_2.txt' 
    test_path     = args.test_path # 'data/mod_test_2.txt' 
    result_path   = args.result_path # 'result/mod/mod_test_2.pred.txt'
    log_path      = args.log_path # 'result/mod_test_2.log'
    save_prefix   = args.save_prefix # 'epochs'
    pretrain_path = args.pretrain_path
    dict_path     = args.dict_path # 'data/mod.txt'

    conv1_kernel = args.conv1_kernel # 10
    conv2_kernel = args.conv2_kernel # 10

    min_rt     = args.min_rt # -60
    max_rt     = args.max_rt # 184
    time_scale = args.time_scale # 60
    max_length = args.max_length # 66

    CNN_EMB    = args.CNN_EMB # True # False # 

    BATCH_SIZE             = args.BATCH_SIZE # 16 # 16
    NUM_CLASSES            = 10
    NUM_EPOCHS             = args.NUM_EPOCHS # 50
    NUM_ROUTING_ITERATIONS = args.NUM_ROUTING_ITERATIONS # 1
    CUDA                   = args.CUDA # True # False #
    LR                     = args.LR # 0.01

    SAVE_MODEL   = args.SAVE_MODEL # False
    EVALUATE_ALL = args.EVALUATE_ALL # True
    
    print(args)
    # ---------- * ---------- * ---------- * ---------- * ---------- * ----------
    if '' == dict_path:
        dict_path = train_path    
    dictionary = Dictionary(dict_path)
    
    param_1D_rt = {'data'            : 'rt',
                   'dim'             : 1,
                   'conv1_kernel'    : (len(dictionary), conv1_kernel),
                   'pri_caps_kernel' : (1, conv2_kernel),
                   'stride'          : 1,
                   'digit_caps_nodes': 32 * 1 * (max_length - conv1_kernel*2 + 2 - conv2_kernel + 1), # 32 # Note: number of conv!
                   'NUM_CLASSES'     : 1
                   }

    param = param_1D_rt

    if 2 == param['dim']:
        print('>> note: using image mode.')
    if 1 == param['dim']:
        print('>> note: using seq mode.')
    return args, train_path, test_path, result_path, log_path, save_prefix, pretrain_path, dict_path, conv1_kernel, conv2_kernel, \
           min_rt, max_rt, time_scale, max_length, CNN_EMB, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, NUM_ROUTING_ITERATIONS, \
           CUDA, LR, SAVE_MODEL, EVALUATE_ALL, dictionary, param

if __name__ == "__main__":    
    args, train_path, test_path, result_path, log_path, save_prefix, pretrain_path, dict_path, conv1_kernel, conv2_kernel, \
    min_rt, max_rt, time_scale, max_length, CNN_EMB, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, NUM_ROUTING_ITERATIONS, \
    CUDA, LR, SAVE_MODEL, EVALUATE_ALL, dictionary, param = param_parse()

    # from torch.autograd import Variable
    from torch.optim import Adam # Adam
    from torchnet.engine import Engine
    # from torchnet.logger import VisdomPlotLogger, VisdomLogger
    # from torchvision.utils import make_grid
    # from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt
    import gc 
    from time import sleep, time
    import timeit
    T1 = timeit.default_timer()

    # read data ========== ========== ========== ========== ========== ==========
    # CNN_EMB = True
    if False == CNN_EMB:
        print('>> note: using one-hot encoding.')
        if True == LOAD_DATA:
            # dictionary = Dictionary(dict_path)
            RTtrain = RTdata(dictionary, 
                             max_length, 
                             train_path,
                             min_rt,
                             max_rt,
                             time_scale)
            RTtest = RTdata(dictionary, 
                            max_length, 
                            test_path,
                            min_rt,
                            max_rt,
                            time_scale)
            with open(RTdata_path, 'wb') as output:
                # pickle.dump(dictionary, output)
                pickle.dump(RTtrain, output)
                pickle.dump(RTtest, output)
        if False == LOAD_DATA:
            with open(RTdata_path, 'rb') as input:
                # dictionary = pickle.load(input)
                RTtrain = pickle.load(input)
                RTtest = pickle.load(input)
                print('>> note: load pre-read RTdata from:', RTdata_path)
        
        # DATA_AUGMENTATION = True
        SPARSE = True
        # def desparse(RTtt):
        #     X = np.zeros((RTtt.number_seq, RTtt.N_aa, RTtt.N_time_step)) # DATA_AUGMENTATION->*2
        #     for i in range(RTtt.number_seq): # DATA_AUGMENTATION->*2
        #         # sparse to dense
        #         X[i,::] = RTtt.X[i].todense()
        #     RTtt.X = X
        if True == SPARSE:
            print('>> note: de-sparse for both train & test data.')
            desparse(RTtrain)
            desparse(RTtest)
    if True == CNN_EMB:
        print('>> note: using >>>embedding<<< method.')
        corpus = Corpus(dictionary, # format: Corpus(dictionary, train_path, val_path='', test_path='', pad_length=0)
                        train_path,
                        test_path=test_path,
                        pad_length=max_length,
                        min_rt=min_rt,
                        max_rt=max_rt,
                        time_scale=time_scale,
                        CLA=args.CLASS or args.regCLASS)

    # read data ========== ========== ========== ========== ========== ==========

    LOG = False
    flog = open(log_path, 'w')

    model = CapsuleNet(param, dictionary,
                       conv1_kernel,
                       conv2_kernel,
                       max_length,
                       args)
    
    if '' == pretrain_path:
        pass
    else:
        model.load_state_dict(torch.load(pretrain_path)) # epoch.pt
        print('>> note: load pre-trained model from:',pretrain_path)

    if True == CUDA:
        model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))
    flog.write("# parameters:"+str(sum(param.numel() for param in model.parameters()))+'\n')

    optimizer = Adam(model.parameters(), lr = LR)
    # optimizer = SGD(model.parameters(), lr = LR/10., momentum = 0.5)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    if 2 == param['dim']:        
        meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)
    if 1 == param['dim']:
        pass
        # meter_mse = tnt.meter.MSEMeter()
    
    if True == LOG:
        train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
        train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
        test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
        test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
        confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
        if 2 == param['dim']:
            ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
            reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss(param, args)
    
    def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')[:47]
        # [torch.ByteTensor of size number x 28 x 28]
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')[:47]
        # [torch.LongTensor of size number]
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    if False == CNN_EMB:
        data_train = torch.FloatTensor(RTtrain.X)
        label_train = torch.FloatTensor(RTtrain.y)
        print('>> note: delete RTtrain.')
        del RTtrain
        gc.collect()
        print('>> sleeping...')
        for i in range(5):
            print('~.~')
        print('>> wake up!')  
    if True == CNN_EMB:
        data_train = corpus.train
        label_train = corpus.train_label
    
    def get_rt_iterator(mode):
        if mode:
            data = data_train # Note: here must be FloatTensor not ByteTensor!            
            labels = label_train
        else:
            if False == CNN_EMB:
                data = torch.FloatTensor(RTtest.X)
                labels = torch.FloatTensor(RTtest.y)
            if True == CNN_EMB:
                data = corpus.test
                labels = corpus.test_label
            # print('>>dim: test data:', data.shape, labels.shape)
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=1, shuffle=mode) # 1 for heatmap
    
    def processor(sample):
        data, labels, training = sample
        # print('>>dim: data, labels, training', data.shape, labels.shape, training)
        # torch.Size([batch, 28, 28]) torch.Size([batch]) True

        if 'mnist' == param['data']:
            data = augmentation(data.unsqueeze(1).float() / 255.0)            
        # print('>>dim: data augmentation', data.shape) # torch.Size([batch, 1, 28, 28])
        # print('>>dim: labels', labels) # Note: labels is already LongTensor?
        if 'rt' == param['data']:
            if False == CNN_EMB:
                data = data[:, None, :, :] # Note: add dimension
            if True == CNN_EMB:
                pass

        if 2 == param['dim']:
            # for classification, we use LongTensor
            labels = torch.LongTensor(labels)
            labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels) 
        if 1 == param['dim']:
            # for regression, we use FloatTensor
            labels = torch.FloatTensor(labels.numpy())
            labels = labels.view(-1, 1) # from [batch] to [batch, 1]

        if True == CUDA:
            data = Variable(data).cuda()
            labels = Variable(labels).cuda()
        if False == CUDA:
            data = Variable(data)
            labels = Variable(labels)

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

    def reset_meters():
        meter_loss.reset()
        if 2 == param['dim']:
            meter_accuracy.reset()            
            confusion_meter.reset()
        if 1 == param['dim']:
            pass
            # meter_mse.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        '''
        So it is just used for recording?
        '''
        if 1 == param['dim']:
            # print('>>dim: state output', state['output'].data.view(-1)) 
            # torch.FloatTensor of size [batch x 10]
            # print('>>dim: state sample', state['sample'][1]) 
            # torch.LongTensor of size [batch]
            # (1): [batch, 1] (2): [batch], so we view (1) as [batch], but no view is fine     
            pass       
            # meter_mse.add(state['output'].data, torch.FloatTensor(state['sample'][1].numpy()))
        if 2 == param['dim']:
            meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
            confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data.cpu())

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        global BEST
        if 2 == param['dim']:
            print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
            flog.write('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)\n' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
            if True == LOG:
                train_loss_logger.log(state['epoch'], meter_loss.value()[0])
                train_error_logger.log(state['epoch'], meter_accuracy.value()[0])
        if 1 == param['dim']:
            print('[Epoch %d] Training Loss: %.4f (MSE)' % (
                state['epoch'], meter_loss.value()[0])) # meter_mse.value()
            flog.write('[Epoch %d] Training Loss: %.4f (MSE)\n' % (
                state['epoch'], meter_loss.value()[0])) # meter_mse.value()
        
        reset_meters()
        
        # iterator
        if 'mnist' == param['data']:
            engine.test(processor, get_iterator(False))
        if 'rt' == param['data']:
            engine.test(processor, get_rt_iterator(False))

        if True == LOG:
            test_loss_logger.log(state['epoch'], meter_loss.value()[0])
            if 2 == param['dim']:
                test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
                confusion_logger.log(confusion_meter.value())
            if 1 == param['dim']:
                test_accuracy_logger.log(state['epoch'], 7) # meter_mse.value()
        
        if 2 == param['dim']:
            print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
            flog.write('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)\n' % (
                state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))
        if 1 == param['dim']:
            print('[Epoch %d] Testing Loss: %.4f (MSE)' % (
                state['epoch'], meter_loss.value()[0])) # meter_mse.value()
            flog.write('[Epoch %d] Testing Loss: %.4f (MSE)\n' % (
                state['epoch'], meter_loss.value()[0])) # meter_mse.value()
        
        if SAVE_MODEL:
            model_save_dir = args.train_path.split('/')[1].split('.')[0]
            if not os.path.isdir(save_prefix+'/'+model_save_dir):
                print('data specific directry has been created:', save_prefix+'/'+model_save_dir)
                os.mkdir(save_prefix+'/'+model_save_dir)            
            if 0 <= state['epoch']: # for heatmap
                torch.save(model.state_dict(), save_prefix+'/'+model_save_dir+'/epoch_%d.pt' % state['epoch'])
                print('>> model: saved.')        
        
        # prediction:
        # model.load_state_dict(torch.load(PATH))
        # pred_data = Variable(torch.FloatTensor(RTtest.X)[:,None,:,:])        
        PRED_BATCH = 17 # 1000 # 16 for heatmap

        if PRED_BATCH > 0:
            '''
            solve memory problem using batch
            '''
            if False == CNN_EMB:
                pred = np.array([])
                # TODO: handle int
                pred_batch_number = int(RTtest.X.shape[0] / PRED_BATCH)+1
                for bi in range(pred_batch_number):
                    test_batch = Variable(torch.FloatTensor(RTtest.X[bi*PRED_BATCH:(bi+1)*PRED_BATCH,:,:])[:,None,:,:])
                    test_batch = test_batch.cuda()
                    pred_batch = model(test_batch)
                    pred = np.append(pred, pred_batch[0].data.cpu().numpy().flatten())
                # print('>>dim: pred', pred.shape)      

                if True == DATA_AUGMENTATION:
                    ''' data augmentation:'''
                    pep_num = int(len(pred) / 2)
                    pred = pred[:pep_num]*0.5 + pred[pep_num:]*0.5
                    obse = RTtest.y[:pep_num]
                    pearson = Pearson(pred,obse)
                    spearman = Spearman(pred,obse)
                else:
                    pearson = Pearson(pred,RTtest.y)
                    spearman = Spearman(pred,RTtest.y)        
            if True == CNN_EMB: # ========== save predictions and features                
                if False:#args.CLASS:
                    pred = np.zeros((0,2))
                    feat = np.zeros((0,2,16))
                else:
                    pred = np.array([])
                    feat = np.zeros((0,16))
                # TODO: handle int
                pred_batch_number = int(corpus.test.shape[0] / PRED_BATCH)+1
                for bi in range(pred_batch_number):
                    test_batch = Variable(corpus.test[bi*PRED_BATCH:(bi+1)*PRED_BATCH,:])
                    test_batch = test_batch.cuda()
                    pred_batch = model(test_batch)
                    if False:#args.CLASS:
                        pred = np.vstack((pred, pred_batch[0].data.cpu().numpy()))
                    else:
                        pred = np.append(pred, pred_batch[0].data.cpu().numpy().flatten())
                    # print(pred_batch[0].shape)
                    # print(pred_batch[1].shape)
                    if EVALUATE_ALL: # also save feature
                        if False:#args.CLASS:
                            feat = np.vstack((feat, pred_batch[1].data.cpu().numpy()))
                        else:    
                            feat = np.vstack((feat, pred_batch[1].view(-1,16).data.cpu().numpy()))                
                if False:#args.CLASS:
                    obse = corpus.test_label.numpy().flatten() 
                    # print(pred,obse)
                    pearson  = accuracy_score(np.argmax(pred,axis=-1), obse)
                    spearman = 0
                elif args.regCLASS or args.CLASS:                    
                    obse = corpus.test_label.numpy().flatten() 
                    # TODO: add metrics
                    predd    = np.array(pred>0.5, dtype=int)
                    TP, FP, TN, FN = perf_measure(obse, predd)
                    SEN = TP/(TP+FN)
                    SPE = TN/(TN+FP)
                    PRE = TP/(TP+FP) if TP+FP != 0 else 0
                    F1  = f1_score(obse, predd)
                    MAT = matthews_corrcoef(obse, predd)
                    ACC = accuracy_score(obse, predd)
                    AUC = roc_auc_score(obse, pred)
                    if ACC > BEST[0] or state['epoch'] == NUM_EPOCHS:
                        if ACC > BEST[0]:
                            BEST = [ACC, state['epoch']]
                        print('~ Best @ Epoch %d, ACC: %.4f' % (BEST[1], BEST[0]))
                else:
                    obse     = corpus.test_label.numpy().flatten() 
                    pearson  = Pearson(pred,obse)
                    spearman = Spearman(pred,obse)                 
                # ========== also get training features
                if EVALUATE_ALL:
                    if False:#args.CLASS:
                        pred_ = np.zeros((0,2))
                        feat_ = np.zeros((0,2,16))
                    else:                        
                        pred_ = np.array([])
                        feat_ = np.zeros((0,16))
                    pred_batch_number = int(corpus.train.shape[0] / PRED_BATCH)+1
                    for bi in range(pred_batch_number):
                        test_batch = Variable(corpus.train[bi*PRED_BATCH:(bi+1)*PRED_BATCH,:])
                        test_batch = test_batch.cuda()
                        pred_batch = model(test_batch)
                        if False:#args.CLASS:
                            pred_ = np.vstack((pred_, pred_batch[0].data.cpu().numpy()))
                            feat_ = np.vstack((feat_, pred_batch[1].data.cpu().numpy()))                                                
                        else:    
                            pred_ = np.append(pred_, pred_batch[0].data.cpu().numpy().flatten())
                            feat_ = np.vstack((feat_, pred_batch[1].view(-1,16).data.cpu().numpy()))                    
        else:
            pred_data = Variable(torch.FloatTensor(RTtest.X)[:,None,:,:]) 
            if True == CUDA:
                pred_data = pred_data.cuda()
            pred = model(pred_data)
            if True == CUDA:
                # print('>>dim: pred', pred[0].data.cpu().numpy().flatten().shape)
                pearson = Pearson(pred[0].data.cpu().numpy().flatten(),RTtest.y)
                spearman = Spearman(pred[0].data.cpu().numpy().flatten(),RTtest.y)
            if False == CUDA:
                pearson = Pearson(pred[0].data.numpy().flatten(),RTtest.y)
                spearman = Spearman(pred[0].data.numpy().flatten(),RTtest.y)
        ''''''
        if args.regCLASS or args.CLASS: 
            print('>> %d testing samples: %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f' %(len(pred), SEN, SPE, PRE, F1, MAT, ACC, AUC))
            flog.write('>> %d testing samples: %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f\n' %(len(pred), SEN, SPE, PRE, F1, MAT, ACC, AUC))
        else:
            print('>> Corr on %d testing samples: %.5f | %.5f' % (len(pred), pearson, spearman))
            flog.write('>> Corr on %d testing samples: %.5f | %.5f\n' % (len(pred), pearson, spearman))
        # writing:
        if True == CNN_EMB:
            obse  = corpus.test_label.numpy().flatten()
            obse_ = corpus.train_label.numpy().flatten()
        if False == CNN_EMB:
            obse = RTtest.y
        
        # ========== write to file
        _result_path = result_path[:-4] +'.'+ str(state['epoch']) + '.txt'        
        with open(_result_path, 'w') as fo: # write prediction
                fo.write('observed\tpredicted\n')
                for i in range(len(pred)):
                    if False:#args.CLASS:
                        fo.write('%.5f\t%.5f\t%.5f\n' % (obse[i], pred[i,0], pred[i,1]))
                    else:
                        fo.write('%.5f\t%.5f\n' % (obse[i], pred[i]))
        if EVALUATE_ALL: # also write test/train features, train pred
            _result_path_ = result_path[:-4] +'.'+ str(state['epoch']) + '_.txt'
            _feat_path    = result_path[:-4] +'.'+ str(state['epoch']) + '.pkl'
            _feat_path_   = result_path[:-4] +'.'+ str(state['epoch']) + '_.pkl'
            with open(_result_path_, 'w') as fo: # write prediction (train set)
                    fo.write('observed\tpredicted\n')
                    for i in range(len(pred_)):
                        if False:#args.CLASS:
                            fo.write('%.5f\t%.5f\t%.5f\n' % (obse_[i], pred_[i,0], pred_[i,1]))
                        else:
                            fo.write('%.5f\t%.5f\n' % (obse_[i], pred_[i]))            
            with open(_feat_path, 'wb') as f: # write testing feature
                pickle.dump(feat, f, pickle.HIGHEST_PROTOCOL)
            with open(_feat_path_, 'wb') as f: # write training feature
                pickle.dump(feat_, f, pickle.HIGHEST_PROTOCOL)
            if state['epoch'] == 1: # only write label at the first epoch
                obse_ = corpus.train_label.numpy().flatten()
                _label_path = result_path[:-8] + 'train.label.txt'
                with open(_label_path, 'wb') as f:
                    pickle.dump(obse_, f, pickle.HIGHEST_PROTOCOL)
        # writing done
        
        # Reconstruction visualization.
        if 2 == param['dim']:
            
            # iterator
            if 'mnist' == param['data']:
                test_sample = next(iter(get_iterator(False)))
            if 'rt' == param['data']:
                test_sample = next(iter(get_rt_iterator(False)))
            # print('>>dim: test_sample', test_sample) # [batch, 28, 28]

            ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
            # print('>>dim: ground_truth', ground_truth.shape) # torch.FloatTensor of size batch x 1 x 28 x 28

            if True == CUDA:
                pred, reconstructions = model(Variable(ground_truth).cuda())
            if False == CUDA:
                pred, reconstructions = model(Variable(ground_truth))
            # print('>>dim: pred', pred)
            reconstruction = reconstructions.cpu().view_as(ground_truth).data

            if True == LOG:
                ground_truth_logger.log(
                    make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
                reconstruction_logger.log(
                    make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    if 'mnist' == param['data']:
        engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
    if 'rt' == param['data']:
        engine.train(processor, get_rt_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)

    T2 = timeit.default_timer()
    print('>> time: %.5f min\n' %((T2-T1)/60.))
    flog.write('>> time: %.5f min\n' %((T2-T1)/60.))
    flog.close()
