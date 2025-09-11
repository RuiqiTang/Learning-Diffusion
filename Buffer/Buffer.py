import random
import torch

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
        items=random.choice(self.buffer,k=n_samples)
        samples,class_ids=zip(*items)

        samples=torch.stack(samples,dim=0)
        class_ids=torch.tensor(class_ids)

        # Put on devices
        samples=samples.to(device)
        class_ids=class_ids.to(device)

        return samples,class_ids
