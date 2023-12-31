#import shutil
import numpy as np
#import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
#import torch.nn.utils.rnn
import h5py
#from torch.autograd import Variable
import time
#from tqdm import tqdm
import sys
import math 
#import platform
import argparse
#import healpy as hp
import ast
import json
import random
#import torch.distributions as td
#from matplotlib.lines import Line2D
#from particle_net import get_model_decay, get_model_TwoBodys, get_model_TwoBodysV2, get_model_TwoBodysV3
from CNNModel import Vgg
#torch.autograd.set_detect_anomaly(True)
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss,self).__init__()
        return
    def forward(self, pred, truth):
        #tmp_sum_pred = torch.FloatTensor([0.])
        #tmp_sum_truth = torch.FloatTensor([0.])
        #tmp_sum_corss = torch.FloatTensor([0.])
        if pred.dim() == 3:
            tmp_sum_pred = torch.sum(pred*pred,(1,2))
            tmp_sum_truth = torch.sum(truth*truth,(1,2))
            tmp_sum_cross = torch.sum(pred*truth,(1,2))
            tmp_loss = (tmp_sum_pred+tmp_sum_truth)/(2*tmp_sum_cross) -1 
            tmp_loss = torch.sum(tmp_loss)/pred.size(0)
            return tmp_loss
        if pred.dim() == 4:
            tmp_sum_pred = torch.sum(pred*pred,(2,3))
            tmp_sum_truth = torch.sum(truth*truth,(2,3))
            tmp_sum_cross = torch.sum(pred*truth,(2,3))
            tmp_loss = (tmp_sum_pred+tmp_sum_truth)/(2*tmp_sum_cross) -1 
            tmp_loss = torch.sum(tmp_loss)/pred.size(0)
            return tmp_loss

        #tmp_loss = (torch.sum(pred*pred) + torch.sum(truth*truth))/(2*torch.sum(pred*truth))
        #tmp_loss0 = (torch.sum(pred[0]*pred[0]) + torch.sum(truth[0]*truth[0]))/(2*torch.sum(pred[0]*truth[0]))
        #print('tmp_loss0=',tmp_loss0-1,'abs sum pred=',torch.sum(torch.abs(pred[0])),',abs sum real=',torch.sum(torch.abs(truth[0])) )
        #return (tmp_loss - 1)/pred.size(0)

def dice_cost(pred_y, label_y):##https://agenda.infn.it/event/28874/contributions/169211/attachments/94397/130957/20220708_AEsforSUEP_ICHEP2022_schhibra.pdf
    tmp_loss = (torch.sum(pred_y*pred_y) + torch.sum(label_y*label_y))/(2*torch.sum(pred_y*label_y))
    return (tmp_loss - 1)/pred_y.size(0)


class link_loss(nn.Module):
    def __init__(self):
        super(link_loss,self).__init__()
        return
    def forward(self,pred_y, label_y, mask):
        tmp_loss = torch.sum( torch.abs((pred_y-label_y)*mask) )/torch.sum(mask)
        return tmp_loss

class loss_2body(nn.Module):
    def __init__(self):
        super(loss_2body,self).__init__()
        return
    def forward(self,pred_y, label_y, mask):##N,P,P N,P,P, N,P,P
        tmp_loss = torch.sum( nn.functional.binary_cross_entropy(input=pred_y, target=label_y,reduction='none')*mask )/torch.sum(mask)
        return tmp_loss


class l1_loss_w(nn.Module):
    def __init__(self):
        super(l1_loss_w,self).__init__()
        return
    def forward(self,pred_y, label_y, weight):
        tmp_loss = torch.sum( torch.abs((pred_y-label_y)*weight) )/pred_y.size(0)
        return tmp_loss

class ce_loss_w(nn.Module):
    def __init__(self):
        super(ce_loss_w,self).__init__()
        self.Loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self,pred_y, label_y, weight):
        tmp_loss = torch.sum(self.Loss(pred_y, label_y)*weight)/pred_y.size(0)
        return tmp_loss





#from torch_geometric.data import Batch as GraphBatch
#from torch_geometric.data import Data as GraphData
#from torch_geometric.loader import DataLoader as GraphDataLoader
#
#def build_graph(filenamelist, device=''):
#    graphs = []
#    for file in filenamelist:
#        f = h5py.File(file, 'r')
#        df = f['feature'][:]
#        for i in range(df.shape[0]):
#            df_i = df[i]##8*25
#            Np = 0
#            for j in range(25):
#                if df_i[0,j]==0 and df_i[1,j]==0:
#                    Np = j
#                    break
#            if Np <=1: continue##at least 2 hits
#            if df_i[1,0] < parsed['E_min']:continue##seed energy
#            dfi = df_i[:,0:Np]
#            dfi[0  ,:] /= 40. ## time scale
#            dfi[2:8,:] /= 100.## position scale
#            dfi = np.transpose(dfi,(1,0))##F,P --> P,F
#            tmp_x = torch.tensor(dfi[1: ,0:8].astype('float32'))
#            tmp_y = torch.tensor(dfi[0:1,0:8].astype('float32'))
#            tb_graph = GraphData(x=tmp_x,y=tmp_y)
#            graphs.append(tb_graph)
#        f.close()
#    #graphs = GraphBatch.from_data_list(graphs).to(device)
#    return graphs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filenamelist):
        super(Dataset, self).__init__()
        print("Reading Dataset")
        self.data = None
        self.data_label = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['feature'][:]##N,13,25
            df_label = f['label'][:]
            #tmp_idx  = df[:,1,0] > parsed['E_min']
            tmp_idx = np.logical_and(df[:,1,parsed['iSeed'],parsed['iSeed']] > parsed['E_min'], df[:,1,parsed['iSeed'],parsed['iSeed']] < parsed['E_max'])
            df = df[tmp_idx]
            df_label = df_label[tmp_idx]
            if df.shape[0] <=0:continue
            df_sub = np.sum(df[:,1,:,:],axis=(1,2),keepdims=False)-df[:,1,parsed['iSeed'],parsed['iSeed']]
            tmp_idx = df_sub > 0
            df = df[tmp_idx]
            df_label = df_label[tmp_idx]
            if df.shape[0] <=0:continue
            df[:,0,:,:] /= 40.
            df_label[:,0] /= parsed['E_scale']
            if parsed['useSeed']:
                #df[:,1,2,2] = 0. ##set seed e to 0
                df[:,1,parsed['iSeed'],parsed['iSeed']] = 0. ##set seed e to 0,FIXME, for 7x7
                #pass
            elif parsed['useAll']:
                pass
            else:
                #df[:,:,2,2] = 0. ##set seed e to 0
                df[:,:,parsed['iSeed'],parsed['iSeed']] = 0. ##set seed e to 0,FIXME, for 7x7
            tmp_label = torch.tensor(df_label.astype('float32'))
            self.data_label = tmp_label if self.data_label is None else torch.cat((self.data_label,tmp_label),0)
            tmp_tensor = torch.tensor(df.astype('float32'))
            self.data = tmp_tensor if self.data is None else torch.cat((self.data,tmp_tensor),0)
            f.close()
                                   
    def __getitem__(self, index):
        da = self.data[index,] if self.data != None else torch.tensor([0])
        da_label = self.data_label[index,] if self.data_label != None else torch.tensor([0])
        return (da, da_label)

    def __len__(self):
        return self.data_label.size()[0]

def count_training_evts(filenamelist):
    tot_n = 0 
    for file in filenamelist:
        f = h5py.File(file, 'r')
        label = f['feature']
        #tot_n += label.shape[0]
        tot_n += np.sum( np.logical_and(label[:,1,parsed['iSeed'],parsed['iSeed']] > parsed['E_min'],label[:,1,parsed['iSeed'],parsed['iSeed']] < parsed['E_max']) )
        #tmp_index = label[:,9]<=0 ##no c14 hit
        #tmp_index1 = label[:,9]>nhit_c14 # c14 hit
        #tot_n_pu += int( np.sum(tmp_index1) )
        f.close()
    return tot_n

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, npe_scale, time_scale, do_log_scale=False):
        super(DatasetTest, self).__init__()
        print("Reading DatasetTest...")
        self.T0 = None
        self.T1 = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['data'][:]
            label = f['label'][:]
            f.close()
            tmp_tensor0 = torch.tensor(label.astype('float32'))
            tmp_tensor1 = torch.tensor(df   .astype('float32'))
            self.T0 = tmp_tensor0 if self.T0 is None else torch.cat((self.T0,tmp_tensor0),0)
            self.T1 = tmp_tensor1 if self.T1 is None else torch.cat((self.T1,tmp_tensor1),0)
        self.T1[:,:,:,0] = self.T1[:,:,:,0]/(1.0*npe_scale)
        self.T1[:,:,:,1] = self.T1[:,:,:,1]/(1.0*time_scale)
        if do_log_scale:
            tmp_t = self.T1[:,:,:,1]
            idx = tmp_t > 0
            tmp_t[idx] = torch.log(tmp_t[idx])
            self.T1[:,:,:,1] = tmp_t
        if channel == 0:#npe
            self.T1 = self.T1[:,:,:,0:1]
        if channel == 1:#ftime
            self.T1 = self.T1[:,:,:,1:2]
        
        self.n = self.T1.size()[0]        
                                    
    def __getitem__(self, index):
        T0 = self.T0[index,]
        T1 = self.T1[index,]
        return (T0,T1)

    def __len__(self):
        return self.n


class DatasetTestTrain(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, npe_scale, time_scale, frac_ep, nhit_c14, do_log_scale=False):
        super(DatasetTestTrain, self).__init__()
        print("Reading DatasetTestTrain")
        self.T = None
        self.Y = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['data'][:]
            label = f['label'][:]
            f.close()
            tmp_ep_index = label[:,9]<=0 ##no c14 hit
            tmp_pu_index = label[:,9]>nhit_c14 # c14 hit

            tmp_tensor = torch.tensor(df[tmp_ep_index,:,:,:].astype('float32'))
            tmp_tensor = tmp_tensor[0:int(tmp_tensor.size(0)*frac_ep)]
            tmp_label = torch.tensor(label[tmp_ep_index].astype('float32'))
            tmp_label = tmp_label[0:int(tmp_label.size(0)*frac_ep)]

            tmp_tensor_pu = torch.tensor(df[tmp_pu_index,:,:,:].astype('float32'))
            tmp_label_pu = torch.tensor(label[tmp_pu_index].astype('float32'))

            tmp_tensor =  torch.cat((tmp_tensor,tmp_tensor_pu),0)
            tmp_label =  torch.cat((tmp_label,tmp_label_pu),0)

            self.T = tmp_tensor if self.T is None else torch.cat((self.T,tmp_tensor),0)
            self.Y = tmp_label if self.Y is None else torch.cat((self.Y,tmp_label),0)
        self.T[:,:,:,0] = self.T[:,:,:,0]/(1.0*npe_scale)
        self.T[:,:,:,1] = self.T[:,:,:,1]/(1.0*time_scale)
        if do_log_scale:
            tmp_t = self.T[:,:,:,1]
            idx = tmp_t > 0
            tmp_t[idx] = torch.log(tmp_t[idx])
            self.T[:,:,:,1] = tmp_t
        if channel == 0:#npe
            self.T = self.T[:,:,:,0:1]
        if channel == 1:#ftime
            self.T = self.T[:,:,:,1:2]

        self.n = self.T.size()[0]        
                                    
    def __getitem__(self, index):
        T1 = self.T[index,]
        T0 = self.Y[index,]
        return (T0,T1)

    def __len__(self):
        return self.n


#def comb_file_block(files_txt,size):
#    files = files_txt.split(';')
#    for file in files:
#        file_block(file,size)


def file_block(files_txt,size):
    blocks = {}
    blocks[0]=[]
    index = 0
    lines = []
    print('files_txt=',files_txt)
    files = files_txt.split(':')
    for file in files:
        print('file=',file)
        if '.txt' not in file:continue
        with open(file,'r') as f:
            tmp_lines = f.readlines()
            for line in tmp_lines:
                if '#' in line:continue
                line = line.replace('\n','')
                line = line.replace(' ','')
                lines.append(line)
    random.shuffle (lines)
    for line in lines:
        if index == size:
            blocks[len(blocks)]=[]
            index = 0
            blocks[int(len(blocks)-1)].append(line)
            index += 1
        else:
            blocks[int(len(blocks)-1)].append(line)
            index += 1
    return blocks


class NN(object):
    def __init__(self, batch_size=64, gpu=0, smooth=0.05, parsed={}):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size        
        self.parsed = parsed
        self.train_file_block = file_block(parsed['train_file'],parsed['train_file_bsize'])
        self.valid_file_block = file_block(parsed['valid_file'],parsed['valid_file_bsize'])
        self.test_file_block  = file_block(parsed['test_file' ],parsed['test_file_bsize'])
        #print(f'train file blocks={self.train_file_block}')
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.input_start = 0
        if self.parsed['notime']:
            self.input_start = 1
        self.input_end = 13
        if self.parsed['usemc']:
            self.input_end += 3

        cfg = {
            'A0': [64,128,256,'M'],
            'A1': [4    , 'M', 16,'M','M'], ##ok
            'A2': [4    , 'M', 16,'M', 32,'M',64,'M'],
            'A3': [16,16,32,32 , 'M', 16,'M', 32,'M',64, 64,'M'], ##good
            'A4': [4    , 'M', 16,'M', 32, 32,'M',64, 64,'M'], 
            'A5': [4,4  , 'M', 16,16,'M', 32, 32,'M',64, 64,'M'], 
            'A31': [4    , 'M', 16,'M', 32,'M',64, 64, 64, 'M'],
            'A32': [4    , 'M', 16,'M', 32,'M',64, 64, 64, 64, 'M'],
            'A33': [4    , 'M', 16,'M', 32,'M',64, 64,'M', 128, 128, 'M'],
            'A': [64    , 'M', 128     , 'M', 256, 256          , 'M', 512, 512          , 'M', 512, 512          , 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256          , 'M', 512, 512          , 'M', 512, 512          , 'M'],
            'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256     , 'M', 512, 512, 512     , 'M'                         ],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256     , 'M', 512, 512, 512     , 'M', 512, 512, 512     , 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        #n_inputs = {'A0':256,'A1':6720,'A2':6272,'A3':2816,'A33':2688,'A':10752,'C':50176,'D':10752}##5x5
        n_inputs = {'A0':1024,'A1':6720,'A2':6272,'A3':2816,'A33':2688,'A':10752,'C':50176,'D':10752}##7x7
        if parsed['useRes']:
            n_inputs = {'A0':1680,'A1':6720,'A2':6272,'A3':5952,'A33':2688,'A':10752,'C':50176,'D':10752}
        
        print('fcs=',parsed['fcs'])
        hyperparameters = {
            'in_channels': int(self.input_end-self.input_start),
            'features_cfg': cfg[parsed['cfg'] ],
            'fcs_cfg':parsed['fcs'],
            'n_input':n_inputs[parsed['cfg'] ],
            'dropout':parsed['Dropout'],
            'Batch_Norm':parsed['BatchNorm'],
            'useRes':parsed['useRes'],
            'bn_after': False,
            'bn_input': False
        }
        self.model = Vgg(hyperparameters).to(self.device)
        print('in_channels=',hyperparameters['in_channels'])
        version_str = torch.__version__ 
        version_tuple = tuple(map(int, version_str.split('.')[:3]))
        if version_tuple > (2,0,0):
            self.model = torch.compile(self.model)
            print('compiled model !')
 
        self.loss = nn.L1Loss()
   
        if parsed['Restore']:
            print('restored from ',parsed['restore_file'])
            checkpoint = torch.load(parsed['restore_file'])
            self.model.load_state_dict(checkpoint['state_dict'])
    def optimize(self, epochs, lr=3e-4):        

        parsed = self.parsed
        print(f'doing optimizing:')
        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = parsed['out_name']

        print(f'Model: {self.out_name}')
        print(" Number of params : ", sum(x.numel() for x in self.model.parameters()))


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if parsed['Restore']:
            print('opt. restored from ',parsed['restore_file'])
            checkpoint = torch.load(parsed['restore_file'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        if parsed['scheduler']=='Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7,patience=2,threshold=0.001,threshold_mode='rel')
        elif parsed['scheduler']=='OneCycleLR':
            total_N = 0
            idx = np.arange(len(self.train_file_block))
            for i in idx:
                tmp_n = count_training_evts(self.train_file_block[i] )
                total_N += tmp_n
            print('tot traning =',total_N)
            total_steps = int(1.0*(total_N)/self.batch_size)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=total_steps, epochs=epochs)

        for epoch in range(1, epochs + 1):
            t_loss, t_corr, t_tot = self.train(epoch)
            v_loss, v_corr, v_tot = self.validate()
            #print(f't_loss={t_loss},t_corr={t_corr},t_tot={t_tot},v_loss={v_loss},v_corr={v_corr},v_tot={v_tot}')
            train_loss = 1.0*t_loss/t_tot
            train_acc  = 1.0*t_corr/t_tot
            valid_loss = 1.0*v_loss/v_tot
            valid_acc  = 1.0*v_corr/v_tot
            current_lr = 0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            print(f'epoch{epoch},train_loss={train_loss},valid_loss={valid_loss}, lr={current_lr}')
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            output_name = self.out_name
            output_name = output_name.replace('.pth','_epoch%d.pth'%epoch)

            #if  (valid_loss < best_loss):
            #if  (train_loss < best_loss):
            if  True:
                best_loss = valid_loss


                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                }
                print("Saving model...")
                torch.save(checkpoint, f'{output_name}')

            if parsed['scheduler']=='Plateau':
                self.scheduler.step(train_loss)
            elif parsed['scheduler']=='StepLR':
                self.scheduler.step()

    def train(self, epoch):
        self.model.train()
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"training Epoch {epoch}/{self.n_epochs}    - t={current_time}")
        idx = np.arange(len(self.train_file_block))
        np.random.shuffle(idx)
        total_loss = 0
        total_corr = 0
        n_total = 0
        for i in idx:
            dataset = Dataset(filenamelist=self.train_file_block[i])
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)
            for batch_idx, (da, da_label) in enumerate(train_loader):
                da = da.to(self.device)   
                da_label = da_label.to(self.device)   
                self.optimizer.zero_grad()
                tmp_list = range(self.input_start,self.input_end)
                z = self.model(da[:,tmp_list,:,:], torch.tensor(0).to(self.device))
                loss = self.loss(input=z.squeeze(), target=da_label[:,0].squeeze())
                loss.backward()
                if self.parsed['clip_grad'] != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.parsed['clip_grad'])
                self.optimizer.step()
                total_loss += loss.item()*z.size(0)
                n_total += z.size(0)
                if parsed['scheduler']=='OneCycleLR': self.scheduler.step()
            
        return (total_loss, total_corr, n_total)

    def validate(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"validing - t={current_time}")
        self.model.eval()
        total_corr = 0
        total_loss = 0
        n_total = 0
        for i in self.valid_file_block:
            dataset = Dataset(filenamelist=self.valid_file_block[i])
            valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)
            with torch.no_grad():
                for batch_idx, (da, da_label) in enumerate(valid_loader):
                    da = da.to(self.device)   
                    da_label = da_label.to(self.device)   
                    tmp_list = range(self.input_start,self.input_end)
                    z = self.model(da[:,tmp_list,:,:], torch.tensor(0).to(self.device))
                    loss = self.loss(input=z.squeeze(), target=da_label[:,0].squeeze())
                    total_loss += loss.item()*z.size(0)
                    n_total += z.size(0)
        return (total_loss, total_corr, n_total)


    def test(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"testing - t={current_time}")
        self.model.eval()
        df_index = 0
        test_i = 0
        data_out = None
        batch_out = None
        for i in self.test_file_block:
            dataset = Dataset(filenamelist=self.test_file_block[i])
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)
            with torch.no_grad():
                for batch_idx, (da, da_label) in enumerate(test_loader):
                    da = da.to(self.device)   
                    da_label = da_label.to(self.device)   
                    tmp_list = range(self.input_start,self.input_end)
                    z = self.model(da[:,tmp_list,:,:], torch.tensor(0).to(self.device))
                    y_pred = z.cpu().detach().numpy()##N,1
                    Y     =  da_label.cpu().detach().numpy()##N,3
                    out  =  np.concatenate((Y,y_pred), axis=1)
                    data_out = out   if data_out is None else np.concatenate((data_out, out  ), axis=0)
                    #batch_out = batch   if batch_out is None else np.concatenate((batch_out, batch  ), axis=0)
        outFile1 = self.parsed['outFile'].replace('.h5','_0.h5')
        hf = h5py.File(outFile1, 'w')
        #hf.create_dataset('batch', data=batch_out)
        hf.create_dataset('data' , data=data_out)
        hf.close()
        print('Saved produced data %s'%outFile1)
    def save_script(self):
        with torch.no_grad():
            device = torch.device("cpu")
            self.model.to(device)
            self.model.eval()
            x, h = torch.rand(26, 8), torch.zeros(26, dtype=torch.long)
            h[25:26] = 1
            print('z0=',self.model(x, h))
            h[15:26] = 1
            print('z1=',self.model(x, h))
            h[0 :25] = 0
            h[25:26] = 1
            ##scripted = torch.jit.script(self.model)
            print('start tracing:')
            scripted = torch.jit.trace(self.model, (x, h))
            print(scripted.code)
            scripted.save(self.parsed['out_ScriptName'])
            print('test loading:')
            loaded = torch.jit.load(self.parsed['out_ScriptName'])
            print('test0:',loaded(x, h))
            h[15:26] = 1
            print('test1:',loaded(x, h))
            return 0           
    #def saveONNX(self):
    #    device = torch.device("cpu")
    #    net.to(device)
    #    net.eval()
    #    tmp_x = torch.randn(1, 6, requires_grad=False)    
    #    torch_out = net(tmp_x)
    #    print('tmp_x=',tmp_x,',torch_out=',torch_out)
    #    netONNX.to(device)
    #    netONNX.eval()
    #    torch_out = netONNX(tmp_x)
    #    print('tmp_x=',tmp_x,',onnx torch_out=',torch_out)
    #    # Export the model
    #    torch.onnx.export(netONNX,               # model being run
    #              tmp_x,                         # model input (or a tuple for multiple inputs)
    #              onnx_file_path,   # where to save the model (can be a file or file-like object)
    #              export_params=True,        # store the trained parameter weights inside the model file
    #              do_constant_folding=True,  # whether to execute constant folding for optimization
    #              input_names = ['input'],   # the model's input names
    #              output_names = ['output'], # the model's output names
    #              dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                            'output': {0 : 'batch_size'}}
    #    )

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int, metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float, metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=200, type=int, metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=128, type=int, metavar='BATCH', help='Batch size')
    parser.add_argument('--train_file', default='', type=str, help='')
    parser.add_argument('--valid_file', default='', type=str, help='')
    parser.add_argument('--test_file' , default='', type=str, help='')
    parser.add_argument('--train_file_bsize', default=150, type=int, help='')
    parser.add_argument('--valid_file_bsize', default=150, type=int, help='')
    parser.add_argument('--test_file_bsize' , default=150, type=int, help='')
    parser.add_argument('--out_name' , default='', type=str, help='')
    parser.add_argument('--channel'  , default=0, type=int, help='0 for npe, 1 for first hit time')
    parser.add_argument('--npe_scale', default=5, type=float, help='')
    parser.add_argument('--time_scale', default=100, type=float, help='')
    parser.add_argument('--do_log_scale', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--scale_1d', default=10., type=float, help='')
    parser.add_argument('--scale_1d_tcor', default=50., type=float, help='')
    parser.add_argument('--E_scale', default=1., type=float, help='')
    parser.add_argument('--R_scale', default=17700., type=float, help='')
    parser.add_argument('--L_scale', default=40000., type=float, help='')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--restore_file' , default='', type=str, help='')
    parser.add_argument('--outFile' , default='', type=str, help='')
    parser.add_argument('--cfg' , default='', type=str, help='')
    parser.add_argument('--BatchNorm', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--Dropout', default=0, type=float, help='')
    parser.add_argument('--DoTest', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--SaveScript', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--out_ScriptName' , default='', type=str, help='')
    parser.add_argument('--DoOptimization', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--clip_grad', default=0, type=float, help='')
    parser.add_argument('--scheduler' , default='StepLR', type=str, help='')
    parser.add_argument('--loss' , default='', type=str, help='')
    parser.add_argument('--ps_features', default=32, type=int, help='')
    parser.add_argument('--ps_input_dropout', default=0.0, type=float, help='')
    parser.add_argument('--psencoding', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--fcs', nargs='+', type=int, help='')
    parser.add_argument('--emb_dim', default=32, type=int, help='')
    parser.add_argument('--n_ext', default=0, type=int, help='')
    parser.add_argument('--activation' , default='relu', type=str, help='')
    parser.add_argument('--weight', default=1., type=float, help='')
    parser.add_argument('--notime', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--usemc', action='store', type=ast.literal_eval, default=False, help='use mc theta, sinphi, cosphi')
    parser.add_argument('--useRes', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--E_min', default=0.0, type=float, help='')
    parser.add_argument('--E_max', default=10.0, type=float, help='')
    parser.add_argument('--useSeed', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--useAll', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--iSeed', default=3, type=int, help='2 for 5x5, 3 for 7x7')
    
    parsed = vars(parser.parse_args())

    network = NN(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'], parsed=parsed)
    if parsed['DoOptimization']:
        network.optimize(parsed['epochs'], lr=parsed['lr'])
    if parsed['DoTest']:
        network.test()
    if parsed['SaveScript']:
        network.save_script()
        #print('self_loss=',self_loss,',l1_loss=',l1_loss)

    #if parsed['saveONNX']:
    #    network.saveONNX()
