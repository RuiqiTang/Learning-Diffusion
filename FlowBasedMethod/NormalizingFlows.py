import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Convolution Normalizing Flows: https://arxiv.org/pdf/1711.02255
'''

class PlanaFlow(nn.Module):
    def __init__(
        self, 
        z_dim:int=2,
        h:function=torch.tanh
    ):
        super().__init__()
        self.z_dim=z_dim
        self.h=h 

        self.u=nn.Parameter(torch.empty(z_dim,1))



class NormalizatingFlow(nn.Module):
    pass