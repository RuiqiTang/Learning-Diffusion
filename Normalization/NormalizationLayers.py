import torch
import torch.nn as nn 
import torch.nn.functional as F

class SpectralNorm:
    '''
        Spectral Normalization
    '''
    def __init__(self,name:str='weight',bound:bool=False,init:bool=True,std:float=1.0):
        self.name=name 
        self.bound=bound
        self.init=init
        self.std=std
    
    def compute_weight(self,module):
        weight=getattr(module,self.name+'_orig')
        u=getattr(module,self.name+'_u')
        size=weight.size()
        weight_mat=weight.continuous().view(size[0],-1)

        with torch.no_grad():
            v=weight_mat.t()@u 
            v=v/v.norm()
            u=weight_mat@v 
            u=u/u.norm()
        
        sigma=u@weight_mat@v 

        if self.bound:
            weight_sn=weight/(sigma+1e-6)*torch.clamp(sigma,max=1)  # for numerical stability
        else:
            weight_sn=weight/sigma
        return weight_sn,u 
    
    def apply(self,module:nn.Module):
        # If: initialization
        if self.init:
            nn.init.normal_(module.weight,0,self.std)
        if hasattr(module,'bias') and module.bias is not None:
            module.bias.data.zero_()

        fn=SpectralNorm(self.name,self.bound)
        weight=getattr(module,self.name)
        del module._parameters[self.name]
        module.register_parameter(self.name+'orig',weight)   # Change registered name
        input_size=weight.size(0)
        u=weight.new_empty(input_size).normal_()

        module.register_buffer(self.name,weight)
        module.register_buffer(self.name+'_u',u)

        module.register_forward_pre_hook(fn)
        return fn 

    def __call__(self,module:nn.Module,input):
        weight_sn,u=self.compute_weight(module)
        setattr(module,self.name,weight_sn)
        setattr(module,self.name+'_u',u)