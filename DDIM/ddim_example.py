import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import torch 
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision import datasets,transforms

from models.ddim import GaussianDiffusion
from models.UNet import UNetModel

batch_size=64
timesteps=500
epochs=10
device='cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()
wandb_api_key=os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

# Load: MNIST dataset
dataset=datasets.MNIST('/Data',train=True,download=True,transform=transform)
train_loader=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

# Load: Model
model=UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1,2,2),
    attention_resolutions=[]
)
model.to(device)

gaussian_diffusion=GaussianDiffusion(timesteps=timesteps)
optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)

# Train Process
images:torch.Tensor
epoch_loss:float

wandb.init(
    project='ddim-mnist',
    config={
        "batch_size":batch_size,
        "timesteps":timesteps,
        "epochs":epochs,
        "lr":5e-4,
        "model_channels":96,
    }
)
wandb.watch(model,log='all')

for epoch in tqdm(range(epochs)):
    epoch_loss=0.0
    for step,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size=images.shape[0]
        images=images.to(device)

        '''
            为每个时间步均匀地采样一个时间步
            每个样本需要在不同的时间步上进行加噪或去噪，通过均匀采样t可以让模型学习到所有时间步的特征，确保对不同程度的噪声都有泛化能力
        '''
        t=torch.randint(0,timesteps,(batch_size,),device=device).long() 
        loss=gaussian_diffusion.train_losses(model,images,t)

        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
        global_step=epoch*len(train_loader)+step
        wandb.log({
            'train/loss':loss.item(),
            'epoch':epoch,
            'step':global_step
        })
    
    avg_loss=epoch_loss/len(train_loader)
    wandb.log({'train/epoch_loss':avg_loss,'epoch':epoch})


# Visualize Sampled Pics
# p_sample
p_sampled_pics=gaussian_diffusion.sample(model,28,batch_size=64,channels=1)
fig=plt.figure(figsize=(12,12),constrained_layout=True)
gs=fig.add_gridspec(8,8)
for n_row in range(8):
    for n_col in range(8):
        f_ax=fig.add_subplot(gs[n_row,n_col])
        image_data=(p_sampled_pics[n_row,n_col]+1.)*255/2
        f_ax.imshow(image_data,cmap='gray')
        f_ax.axis('off')
wandb.log({'generaed_samples/p_sample':wandb.Image(fig)})
# DDIM sample
ddim_sampled_pics=gaussian_diffusion.ddim_sample(model,28,batch_size=64,channels=1)
fig=plt.figure(figsize=(12,12),constrained_layout=True)
gs=fig.add_gridspec(8,8)
for n_row in range(8):
    for n_col in range(8):
        f_ax=fig.add_subplot(gs[n_row,n_col])
        image_data=(ddim_sampled_pics[n_row,n_col]+1.)*255/2
        f_ax.imshow(image_data,cmap='gray')
        f_ax.axis('off')
wandb.log({'generaed_samples/ddim_sample':wandb.Image(fig)})

wandb.finish()


        



