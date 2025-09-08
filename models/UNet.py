import torch 
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import math 
from typing import Tuple

def timestep_embedding(timesteps:int,dim:int,max_period:int=10000):
    '''
        shape of timesteps: [N]
        shape of return: [N,dim]
    '''
    half_dim=dim//2
    freqs=torch.exp(-torch.log(max_period)*\
                   torch.arange(start=0,end=half_dim,dtype=torch.float32)/half_dim
        ).to(device=timesteps.device)
    phase=timesteps[:,None].float()*freqs[None]
    embedding=torch.cat([torch.cos(phase),torch.sin(phase)],dim=-1)
    if dim%2:
        embedding=torch.cat([embedding,torch.zeros_like(embedding[:,:1])],dim=-1)
    return  embedding

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self,x,emb):
        return

class TimestepEmbedSequential(nn.Sequential,TimestepBlock):
    def forward(self, x,emb):
        # Check Input
        for layer in self:
            if isinstance(layer,TimestepBlock):
                x=layer(x,emb)
            else:
                x=layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int,
        out_channels:int,
        time_channels:int,
        dropout:float
    ):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.GroupNorm(32,in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

        # Time Embedding
        self.time_emb=nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=time_channels,
                out_features=out_channels
            )
        )

        self.conv2=nn.Sequential(
            nn.GroupNorm(32,out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

        if in_channels!=out_channels:
            self.shortcut=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        else:
            self.shortcut=nn.Identity() # output=input
    
    def forward(self,x,t):
        '''
            shape of x: [batch_size,in_dim,height,width]
            shape of t: [batch_size,time_dim]
        '''
        h=self.conv1(x)
        h+=self.time_emb(t)[:,:,None,None]  # Expand to the same dimension as x
        h=self.conv2(h)
        return  h+self.shortcut(x)

class AttentionBlock(nn.Module):
    '''
        Attention Block with shortcut
    '''
    def __init__(
        self,
        channels:int,
        num_heads:int=1
    ):
        super().__init__()
        self.num_heads=num_heads
        assert channels%num_heads==0

        self.norm_layer=nn.GroupNorm(32,channels)
        self.qkv=nn.Conv2d(
            in_channels=channels,
            out_channels=channels*3,
            kernel_size=1,  # 1x1 conv的作用是对每个空间位置的通道维度进行线性变换
            bias=False
        )
        self.proj=nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )
    
    def forward(self,x):
        B,C,H,W=x.shape
        qkv=self.qkv(x)

        h_dim=C//self.num_heads
        q,k,v=qkv.reshape(B*self.num_heads,3*h_dim,H*W).chunk(3,dim=1)
        
        scale=1./math.sqrt(math.sqrt(h_dim))
        attn=torch.einsum('bct,bcs->bts',q*scale,k*scale).softmax(dim=-1)
        h=torch.einsum('bts,bcs->bts',attn,v).reshape(B,-1,H,W)
        h=self.proj(h)
        return  h+x

class UpSample(nn.Module):
    def __init__(
        self,
        channels:int,
        use_conv:bool=True
    ):
        super().__init__()
        self.use_conv=use_conv
        if self.use_conv:
            self.conv=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            )
    
    def forward(self,x):
        x=F.interpolate(
            x,
            scale_factor=2,
            mode='nearest'
        )   #Expand to 2x
        if self.use_conv:
            x=self.conv(x)
        return x

class DownSample(nn.Module):
    def __init__(
            self,
            channels:int,
            use_conv:bool=True
        ):
        super().__init__()
        if use_conv:
            self.operation=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        else:
            self.operation=nn.AvgPool2d(stride=2)
    
    def forward(self,x):
        return self.operation(x)

class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels:int=3,
        out_channels:int=3,
        model_channels:int=128,
        num_res_block:int=2,
        attention_resolution:Tuple=(8,16),
        dropout:float=0.,
        channel_mult:Tuple=(1,2,2,3),
        conv_resample:bool=True,
        num_heads:int=4
    ):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.model_channels=model_channels


        # Time Embedding
        time_emb_dim=self.model_channels*4
        self.time_embed=nn.Sequential(
            nn.Linear(in_features=model_channels,out_features=time_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dim,out_features=time_emb_dim)
        )

        self.down_blocks=nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=model_channels,
                    kernel_size=3,
                    padding=1
                ))
            ]
        )

        # Construct Down Blocks
        down_blocks_channels=[model_channels]
        ch=model_channels
        ds=1
        for level,mult in enumerate(channel_mult):
            for _ in range(num_res_block):
                layers=[
                    ResidualBlock(
                        in_channels=ch,out_channels=mult*model_channels,time_channels=time_emb_dim,dropout=dropout
                    )
                ]
                ch=mult*model_channels
                if ds in attention_resolution:
                    layers.append(AttentionBlock(channels=ch,num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_blocks_channels.append(ch)
            if level!=len(channel_mult)-1:
                self.down_blocks.append(
                    TimestepEmbedSequential(DownSample(channels=ch,use_conv=conv_resample))
                )
                down_blocks_channels.append(ch)
                ds*=2
        
        # Construct Middle Blocks
        self.middle_block=TimestepEmbedSequential(
            ResidualBlock(in_channels=ch,out_channels=ch,time_channels=time_emb_dim,dropout=dropout),
            AttentionBlock(channels=ch,num_heads=num_heads),
            ResidualBlock(in_channels=ch,out_channels=ch,time_channels=time_emb_dim,dropout=dropout)
        )

        # Construct Up Blocks
        self.up_blocks=nn.ModuleList([])
        for level,mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_block+1):
                layers=[
                    ResidualBlock(in_channels=ch+down_blocks_channels.pop(),
                                  out_channels=model_channels*mult,
                                  time_channels=time_emb_dim,
                                  dropout=dropout)
                ]
                ch=model_channels*mult
                if ds in attention_resolution:
                    layers.append(AttentionBlock(channels=ch,num_heads=num_heads))
                if level and i==num_res_block:
                    layers.append(UpSample(channels=ch,use_conv=conv_resample))
                    ds//=2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out=nn.Sequential(
            nn.GroupNorm(32,ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=model_channels,out_channels=out_channels,kernel_size=3,padding=1)
        )
    
    def forward(self,x,timesteps):
        hs=[]
        t_emb=self.time_embed(timestep_embedding(timesteps=timesteps,dim=self.model_channels))

        # Down stage
        h=x
        for module in self.down_blocks:
            h=module(x,t_emb)
            hs.append(h)
        
        # Middle stage
        h=self.middle_block(h,t_emb)

        # Up stage
        for module in self.up_blocks:
            cat_in=torch.cat([h,hs.pop()],dim=-1)
            h=module(cat_in,t_emb)
        return  self.out(h)

