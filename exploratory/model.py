import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class PositionEncoder(nn.Module): 
    '''
        Encodes the position of the words into sin's and cos's
    '''
    def __init__(self,ntoken,dmodel):
        super(PositionEncoder,self).__init__()



def word2vec(x:str)->list:
    '''
       Takes a word sequence and convert it to a vector embedding
    '''
    pass 

class SmallTransformer(nn.Transformer):
    def __init__(self,ntoken,ninp,nhead,nhid,nlayers,dropout=0.3):
        super(SmallTransformer,self).__init__(d_model=ninp,nhead=nhead, 
                                              dim_feedforward=nhid, num_encoder_layers=nlayers)
        

    def forward(self,x): 
        pass 
