import random
import torch
from typing import Tuple
import numpy as np

class SampleBuffer:
    def __init__(self,max_samples:int=10000):
        self.max_samples=max_samples
        self.buffer=[]
    
    def __len__(self):
        return len(self.buffer)

    def push(self,samples,class_ids=None):
        samples=samples.detach().to('cpu')
        class_ids=class_ids.detach().to('cpu')

        for sample,class_id in zip(samples,class_ids):
            self.buffer.append((sample.detach(),class_id))
            if len(self.buffer)>self.max_samples:
                self.buffer.pop(0)
    
    def get(self,n_samples:int,device='cuda'):
        # Randomly select sample
        items=random.choices(self.buffer,k=n_samples)
        samples,class_ids=zip(*items)

        samples=torch.stack(samples,dim=0)
        class_ids=torch.tensor(class_ids)

        # Put on devices
        samples=samples.to(device)
        class_ids=class_ids.to(device)

        return samples,class_ids

    def sample(self,size:Tuple[int,int,int],batch_size:int=128,p:float=.95,device='cuda'):
        r'''
            size:[ch,img_size,max_class]
        '''
        return  self.hybrid_sample(
            size=size,
            batch_size=batch_size,
            p=p,
            device=device
        )
    
    def hybrid_sample(self,size:Tuple[int,int,int],batch_size:int=128,p:float=.95,device='cuda'):
        '''
            :param p: propotion for sampling
        '''
        channels:int 
        img_size:int 
        max_classes:int
        channels,img_size,max_classes=size

        # If: no samples
        if len(self.buffer)<1:
            return  (
                torch.randn(batch_size,channels,img_size,img_size,device=device),
                torch.randint(0,max_classes,(batch_size,),device=device),
            )

        n_reply=(np.random.randn(batch_size)<p).sum()

        reply_sample,reply_id=self.get(n_reply)
        random_samples=torch.rand(batch_size-n_reply,channels,img_size,img_size,device=device)
        random_id=torch.randint(0,max_classes,(batch_size-n_reply,),device=device)
        return  (
            torch.cat([reply_sample,random_samples],0),
            torch.cat([reply_id,random_id],0),
        )
