import os 
from tqdm import tqdm 
from dotenv import load_dotenv

import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms

from models.IGEBM import IGEBM
from utils.utils import sample_data,clip_grad
from utils.Buffer import SampleBuffer

# Init Wandb
import wandb
from dotenv import load_dotenv
load_dotenv()
wandb_api_key=os.getenv('WANDB_API_KEY')
# wandb.login(key=wandb_api_key)

# Set: parameters
batch_size=64
lr=1e-4
max_classes=10
sample_step=60
step_size=10
alpha=1
device='cuda' if torch.cuda.is_available() else 'cpu'
# wandb.init(project='mnist-igebm')

# Load: MNIST dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
dataset=datasets.MNIST('/Data',train=True,download=True,transform=transform)
loader=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)
# Obtain one sample, deduce shapes
first_batch=next(iter(loader))
imgs,_=first_batch
channels=imgs.shape[1]
img_size=imgs.shape[2]

loader=tqdm(enumerate(sample_data(loader)))

buffer=SampleBuffer()
noise=torch.randn(batch_size,channels,img_size,img_size,device=device)

# Set models and optimizers
model=IGEBM(n_class=max_classes)
model=model.to(device)
parameters=model.parameters()
optimizer=optim.Adam(parameters,lr=lr,betas=(0.,.999))

# Train & Eval process
pos_img:torch.Tensor
pos_id:torch.Tensor
for i,(pos_img,pos_id) in loader:
    pos_img,pos_id=pos_img.to(device),pos_id.to(device)

    # Generate negative samples
    neg_img,neg_id=buffer.sample(size=[channels,img_size,max_classes],
                                 batch_size=pos_img.shape[0])
    neg_img.requires_grad=True
    for p in parameters:
        p.requires_grad=False
    model.eval()

    for k in tqdm(range(sample_step)):
        if noise.shape[0]!=neg_img.shape[0]:
            noise=torch.randn(neg_img.shape[0],channels,img_size,img_size,device=device)
        noise.normal_(0.0,.005) 
        neg_img.data.add_(noise.data)

        neg_out:torch.Tensor=model(neg_img,neg_id)
        neg_out.sum().backward()
        neg_img.grad.data.clamp(-.01,.01)   # get gradient

        '''
            $\tilde{x}^k\leftarrow \tilde{x}^{k-1}-\nabla_x E_\theta(\tilde{x}^{k-1})+\omega$
        '''
        neg_img.data.add_(-step_size,neg_img.grad.data)
        neg_img.grad.detach_()
        neg_img.grad.zero_()
        neg_img.data.clamp_(0,1)

    neg_img=neg_img.detach()

    # Train process
    for p in parameters:
        p.requires_grad=True
    model.train()
    model.zero_grad()

    pos_out=model(pos_img,pos_id)
    neg_out=model(neg_img,neg_id)

    '''
        Construct Loss function

        $\mathcal{L}=\frac{1}{N}\sum_i \alpha(E_\theta(x_i^+)^2+E_\theta(x_i^-)^2)+(E_\theta(x_i^+)-E_\theta(x_i^-))$

    '''
    loss=((pos_out**2+neg_out**2)*alpha+(pos_out-neg_out)).mean()
    loss.backward()
    clip_grad(parameters,optimizer)
    optimizer.step()

    buffer.push(neg_img,neg_id)
    # print(i,loss.item())
    # wandb.log({'loss':loss.item(),'step':i})

    if i%100==0:
        pos_to_show=pos_img[:4].detach().cpu()
        neg_to_show=neg_img[:4].detach().cpu()

        # wandb.log({
        #     "pos_img": [wandb.Image(img, caption=f"pos_{j}") for j, img in enumerate(pos_to_show)],
        #     "neg_img": [wandb.Image(img, caption=f"neg_{j}") for j, img in enumerate(neg_to_show)],
        #     "step": i
        # })
