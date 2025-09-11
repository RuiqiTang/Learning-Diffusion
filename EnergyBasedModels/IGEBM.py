import torch
import torch.nn as nn 
import torch.nn.functional as F

from ..Normalization.NormalizationLayers import SpectralNorm

class ResBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,n_class=None,downsample=False):
        super().__init__()

        self.conv1=SpectralNorm().apply(
            nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False if n_class is not None else True)
        )
        self.conv2=SpectralNorm(std=1e-10,bound=True).apply(
            nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False if n_class is not None else True)
        )   

        self.class_embed=None 

        if n_class is not None:
            class_emb=nn.Embedding(n_class,out_channels*2*2)
            class_emb.weight.data[:,:out_channels*2]=1
            class_emb.weight.data[:,out_channels*2:]=0
            self.class_embed=class_emb
        
        self.skip=None
        if in_channels!=out_channels or downsample:
            self.skip=nn.Sequential(
                SpectralNorm().apply(nn.Conv2d(in_channels,out_channels,1,bias=False))
            )
        self.downsample=downsample
    
    def forward(self,input,class_id=None):
        x=input
        x=self.conv1(x)
        if self.class_embed is not None:
            emb=self.class_embed(class_id).view(input.shape[0],-1,1,1)
            w1,w2,b1,b2=emb.chunk(4,1)
            out=w1*x+b1 
        
        x=F.leaky_relu(x,negative_slope=.2)
        x=self.conv2(x)

        if self.class_embed is not None:
            x=w2*x+b2 
        if self.skip is not None:
            skip=self.skip(x)
        else:
            skip=skip
        
        x=x+skip
        if self.downsample:
            x=F.avg_pool2d(x,2)
        x=F.leaky_relu(x,negative_slope=.2)
        return x