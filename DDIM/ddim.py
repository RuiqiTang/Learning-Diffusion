import torch
import math
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def linear_beta_schedule(timesteps:int):
    scale=1000/timesteps
    beta_start=scale*1e-4
    beta_end=scale*0.02
    return  torch.linspace(beta_start,beta_end,timesteps,dtype=torch.float64)

def cosine_beta_schedule(timesteps:int,s:float=0.008):
    # REF: https://arxiv.org/abs/2102.09672
    steps=timesteps+1
    x=torch.linspace(0,timesteps,steps,dtype=torch.float64)
    alphas_cumprod=torch.cos(((x/timesteps+s)/(1+s)*math.pi*0.5)**2)
    alphas_cumprod=alphas_cumprod/alphas_cumprod[0]
    betas=1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return  torch.clip(betas,0,0.999)

class GaussianDiffusion:
    def __init__(self,timesteps:int=1000,beta_schedule:str='linear'):
        self.timesteps=timesteps

        # Set beta
        if beta_schedule=='linear':
            betas=linear_beta_schedule(timesteps)
        elif beta_schedule=='cosine':
            betas=cosine_beta_schedule(timesteps)
        self.betas=betas

        self.alphas=1.-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,axis=0)
        self.alphas_cumprod_prev=F.pad(self.alphas_cumprod[:-1],(1,0),value=1.)

        # Calculation for diffusion q(x_t|x_{t-1})
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod=torch.log(1.-self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod=torch.sqrt(1./self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod=torch.sqrt(1./self.alphas_cumprod-1)

        # Calculation for diffusion q(x_{t-1}|x_t,x_0)
        self.posterior_variance=(
            self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped=torch.log(
            torch.cat([self.posterior_variance[1:2],self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1=(
            self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.-self.alphas_cumprod)
        )
        self.posterior_mean_coef2=(
            (1.-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.-self.alphas_cumprod)
        )

    def _extract(self,a,t,x_shape):
        batch_size=t.shape[0]
        out=a.to(t.device).gather(0,t).float()  # Extarct a[t,:]
        out=out.reshape(batch_size,*((1,)*(len(x_shape)-1)))    # Reshape, in order to broadcast with x_shape
        return  out 
    
    # Forward Diffusion: q(x_t|x_0)
    def q_sample(self,x_start:torch.Tensor,t,noise:torch.Tensor=None):
        if noise is None:
            noise=torch.randn_like(x_start)
        sqrt_alphas_cumprod_t=self._extract(self.sqrt_alphas_cumprod,t,x_start.shape)
        sqrt_one_minus_alpha_cumprod_t=self._extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)
        return  sqrt_alphas_cumprod_t*x_start+sqrt_one_minus_alpha_cumprod_t*noise
    
    # Mean and Var of q(x_t|x_0)
    def q_mean_variance(self,x_start:torch.Tensor,t):
        mean=self._extract(self.sqrt_alphas_cumprod,t,x_start.shape)*x_start
        var=self._extract(1.-self.alphas_cumprod,t,x_start.shape)
        log_var=self._extract(self.log_one_minus_alphas_cumprod,t,x_start.shape)
        return  mean,var,log_var
    
    # Mean and Var of q(x_{t-1}|x_t,x_0)
    def q_posterior_mean_variance(self,x_start:torch.Tensor,x_t:torch.Tensor,t):
        post_mean=(
            self._extract(self.posterior_mean_coef1,t,x_t.shape)*x_start+\
            self._extract(self.posterior_mean_coef2,t,x_t.shape)*x_t
        )
        post_var=self._extract(self.posterior_variance,t,x_t.shape)
        pos_log_var_clipped=self._extract(self.posterior_log_variance_clipped,t,x_t.shape)
        return post_mean,post_var,pos_log_var_clipped

    # Compute x_0 from x_t and pred noise
    def predict_start_from_noise(self,x_t:torch.Tensor,t,noise:torch.Tensor):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod,t,x_t.shape)*x_t-
            self._extract(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape)*noise
        )

    # Mean and Var of p(x_{t-1}|x_t)
    def p_mean_variance(self,model:torch.nn,x_t:torch.Tensor,t,clip_denoised:bool=True):
        pred_noise=model(x_t,t)
        x_recon=self.predict_start_from_noise(x_t,t,pred_noise)
        if clip_denoised:
            x_recon=torch.clamp(x_recon,min=-1,max=1.)
        model_mean,post_var,post_log_var=self.q_posterior_mean_variance(x_recon,x_t,t)
        return  model_mean,post_var,post_log_var
    
    # Sample from p(x_{t-1}|x_t), Revealing Process
    @torch.no_grad()
    def p_sample(self,model,x_t:torch.Tensor,t,clip_denoised:bool=True):
        model_mean,_,model_log_var=self.p_mean_variance(
            model,x_t,t,clip_denoised
        )
        noise=torch.rand_like(x_t)
        nonzero_mask=((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))

        # Cal: x_{t-1}
        pred_img=model_mean+nonzero_mask*(0.5*model_log_var).exp()*noise
        return  pred_img
    
    # Reverse Diffusion
    @torch.no_grad()
    def p_sample_loop(self,model,shape):
        batch_size=shape[0]
        device=next(model.parameters()).device
        img=torch.randn(shape,device=device)    # Start from noise
        img_list=[]
        for i in tqdm(reversed(range(0,self.timesteps)),desc="Sampling Loop"):
            img=self.p_sample(model,img,torch.full((batch_size,), i, device=device, dtype=torch.long))
            img_list.append(img.cpu().numpy())
        return  img_list
    
    @torch.no_grad()
    def sample(self,model,image_size,batch_size:int=8,channels:int=3):
        return  self.p_sample_loop(model,shape=(batch_size,channels,image_size,image_size))
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_size,
        batch_size:int=3,
        channels:int=3,
        ddim_timesteps:int=50,
        ddim_discr_method:str='uniform',
        ddim_eta:float=0.0,
        clip_denoised:bool=True
    ):
        # Set: DDIM Timestamp Seq
        if ddim_discr_method=='uniform':
            c=self.timesteps//ddim_timesteps
            ddim_timestep_seq=np.asarray(list(range(0,self.timesteps,c)))
        elif ddim_discr_method=='quad':
            ddim_timestep_seq=(
                np.linspace(0,np.sqrt(self.timesteps*.8,ddim_timesteps))**2
            ).astype(int)
        
        ddim_timestep_seq+=1
        ddim_timestep_prev_seq=np.append(np.array([0]),ddim_timestep_seq[:-1])

        device=next(model.parameters()).device
        sample_img=torch.randn((batch_size,channels,image_size,image_size),device=device)
        for i in tqdm(reversed(range(0,ddim_timesteps)),desc="Sampling Loop(DDIM)"):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # Extract: alpha_cumprod , prev_alpha_cumprod
            alpha_cumprod_t=self._extract(self.alphas_cumprod,t,sample_img.shape)
            alpha_cumprod_t_prev=self._extract(self.alphas_cumprod,prev_t,sample_img.shape)

            # Predict: Noise
            pred_noise=model(sample_img,t)

            # Predict: Start x_0
            pred_x0=(sample_img-torch.sqrt(1.-alpha_cumprod_t)*pred_noise)/torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                prev_x0=torch.clamp(pred_x0,min=-1.,max=1.)
            
            '''
                Compute Variance: $\sigma_t(\eta)$

                $\sigma_t=\sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\times \sqrt{1-\frac{\alpha_t}{\alpha_{t-1}}}$

            '''
            sigmas_t=ddim_eta*torch.sqrt((1.-alpha_cumprod_t_prev)/(1.-alpha_cumprod_t)*(1-alpha_cumprod_t/alpha_cumprod_t_prev))

            '''
                generate sample $x_{t-1}$ from a sample $x_t$

                $x_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}\right)+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\epsilon_\theta^{(t)}(x_t)+\sigma_t\epsilon_t$

            '''
            dir_point_to_x_t=torch.sqrt(1.-alpha_cumprod_t_prev-sigmas_t**2)*pred_noise
            x_prev=torch.sqrt(alpha_cumprod_t_prev)*prev_x0+dir_point_to_x_t+sigmas_t*torch.randn_like(sample_img)

            sample_img=x_prev
        return  sample_img.cup().numpy()
    
    def train_losses(self,model,x_start:torch.Tensor,t):
        noise=torch.randn_like(x_start)
        x_noised=self.q_sample(x_start,t,noise)
        predicted_noise=model(x_noised,t)
        loss=F.mse_loss(noise,predicted_noise)
        return  loss
