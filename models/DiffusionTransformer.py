'''
    REFERENCE: https://github.com/facebookresearch/DiT
'''

import torch
import torch.nn as nn
import numpy as np
import math 
from timm.models.vision_transformer import PatchEmbed,Attention,Mlp 

def modulate(x,shift,scale):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class TimeStepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size:int,
        freq_emb_size:int=256
    ):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=freq_emb_size,out_features=hidden_size,bias=True),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
        )
        self.freq_emb_size=freq_emb_size
    
    @staticmethod
    def timestep_embedding(timesteps:torch.Tensor,dim:int,max_period:int=10000):
        '''
            shape of timesteps: [N]
            shape of return: [N,dim]
        '''
        half_dim=dim//2
        freqs=torch.exp(-torch.log(max_period)*torch.arange(start=0,end=half_dim,dtype=torch.float32)/half_dim).to(device=timesteps.device)
        phase=timesteps[:,None].float()*freqs[None]
        embedding=torch.cat([torch.cos(phase),torch.sin(phase)],dim=-1)
        if dim%2:
            embedding=torch.cat([embedding,torch.zeros_like(embedding[:,:1])],dim=-1)
        return  embedding

    def forward(self,timestep:torch.Tensor):
        t_freq=self.timestep_embedding(timesteps=timestep,dim=self.freq_emb_size)
        t_emb=self.mlp(t_freq)
        return  t_emb

class LabelEmbedder(nn.Module):
    '''
        Introduced Dropout Layer in order to use Classifier-Free Guidence Efficiently
    '''
    def __init__(
            self,
            num_classes:int,
            hidden_size:int,
            dropout_prob:float   
    ):
        super().__init__()
        use_cfg_embedding=dropout_prob>0
        self.embedding_table=nn.Embedding(num_embeddings=num_classes+use_cfg_embedding,embedding_dim=hidden_size)
        self.num_classes=num_classes
        self.dropout=dropout_prob
    
    def token_drop(self,labels:torch.Tensor,force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids=torch.rand(labels.shape[0],device=labels.device)<self.dropout
        else:
            drop_ids=force_drop_ids==1
        labels=torch.where(drop_ids,self.num_classes,labels)
        return  labels

    def forward(self,labels:torch.Tensor,train:bool,force_drop_ids=None):
        use_dropout=self.dropout>0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels=self.token_drop(labels=labels,force_drop_ids=force_drop_ids)
        embeddings=self.embedding_table(labels)
        return  embeddings

class DiTBlock(nn.modules):
    def __init__(self,hidden_size:int,num_heads:int,mlp_ratio:float=4.0,**block_kwargs):
        super().__init__()
        self.norm1=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attn=Attention(hidden_size,num_heads=num_heads,qkv_bias=True,**block_kwargs)
        self.norm2=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)

        mlp_hidden_dim=int(mlp_ratio*hidden_size)
        approx_gelu=lambda: nn.GELU(approximate='tahn')
        self.mlp=Mlp(in_features=hidden_size,hidden_features=mlp_hidden_dim,act_layer=approx_gelu,drop=0.)

        self.adaLN_modulation=nn.Sequential(    #从条件信息c中产生6组调制参数
            nn.SiLU(),
            nn.Linear(hidden_size,6*hidden_size,bias=True)
        )

    def forward(self,x,c):
        '''
            :param c: conditional information
        '''
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.adaLN_modulation(c).chunk(6,dim=1)
        x=x+gate_msa.unsqueeze(1)*self.attn(modulate(self.norm1(x),shift=shift_msa,scale=scale_msa))
        x=x+gate_mlp.unsqueeze(1)*self.attn(modulate(self.norm2(x),shift=shift_mlp,scale=scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self,hidden_size:int,patch_size:int,out_channels:int):
        super().__init__()
        self.norm_final=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.linear=nn.Linear(hidden_size,patch_size*patch_size*out_channels,bias=True)
        self.adaLN_modulation=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,2*hidden_size,bias=True)
        )
    
    def forward(self,x,c):
        shift,scale=self.adaLN_modulation(c).chunk(2,dim=1)
        x=modulate(self.norm_final(x),shift=shift,scale=scale)
        x=self.linear(x)
        return x 

class DiT(nn.Module):
    def __init__(
            self,
            input_dim:int=32,
            patch_size:int=2,
            in_channels:int=4,
            hidden_size:int=1152,
            depth:int=28,
            num_heads:int=16,
            mlp_ratio:float=4.0,
            class_dropout_prob:float=0.1,
            num_classes:int=1000,
            learn_sigma:bool=True
        ):
        super().__init__()
        self.learn_sigma=learn_sigma
        self.in_channels=in_channels
        self.out_channels=in_channels*2 if learn_sigma else in_channels
        self.patch_size=patch_size
        self.num_heads=num_heads

        self.x_embedder=PatchEmbed(input_dim,patch_size,in_channels,hidden_size,bias=True)
        self.t_embedder=TimeStepEmbedder(hidden_size)
        self.y_embedder=LabelEmbedder(num_classes,hidden_size,class_dropout_prob)
        num_patches=self.x_embedder.num_patches
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches,hidden_size),requires_grad=False)

        self.blocks=nn.ModuleList([
            DiTBlock(hidden_size,num_heads,mlp_ratio) for _ in range(depth)
        ])
        self.final_layer=FinalLayer(hidden_size,patch_size,self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module,nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias,0)
        
        self.apply(_basic_init)

        # Init: pos_emb

        # Init: patch_emb
        w=self.x_embedder.proj.weight.data
        nn.init.xavier_normal_(w.view([w.shape[0],-1]))
        nn.init.constant_(self.x_embedder.proj.bias,0)

        # Init: label_emb
        nn.init.normal_(self.y_embedder.embedding_table.weight,std=0.02)

        # Init: time_emb
        nn.init.normal_(self.t_embedder.mlp[0].weight,std=0.02)
        nn.init.normal_(self.t_embedder.mlp[1].weight,std=0.02)

        # Init: adaLN_modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight,0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,0)
        
        # Init: Output Layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight,0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,0)
        nn.init.constant_(self.final_layer.linear.weight,0)
        nn.init.constant_(self.final_layer.linear.bias,0)
    
    def unpatchify(self,x:torch.Tensor):
        '''
            :param x:
                - shape: [N,T,patch_size^2*C]
                    - N: batch_size
                    - T: num of patches, T=H*W
                    - patch_size
                    - C: channels
            :return:
                - shape: [N,C,H,W]
        '''
        c=self.out_channels
        p=self.x_embedder.patch_size[0]
        h=w=int(x.shape[1]**0.5)
        assert h*w==x.shape[1]

        x=x.reshape(shape=(x.shape[0],h,w,p,p,c))
        x=torch.einsum('nhwpqc->nchpwq',x)  # Exchange dimensions
        imgs=x.reshape(shape=(x.shape[0],c,h*p,w*p))
        return  imgs

    def forward(self,x:torch.Tensor,t:torch.Tensor,y:torch.Tensor):
        '''Forward Process of DiT
            :param x:
                - shape: [N,C,H,W]
            :param y:
                - shape: [N]
            :param t:
                - shape: [N]
        '''
        x=self.x_embedder(x)+self.pos_embed(x)  # [N,T,D]
        t=self.t_embedder(t)    # [N,D]
        y=self.y_embedder(y,self.training)  # [N,D]
        c=t+y 

        for block in self.blocks:
            x=block(x,c)    #[N,T,D]
        x=self.final_layer(x,c)
        x=self.unpatchify(x)    #[N,C,H,W]
        return x


