from torch.utils.data import DataLoader
import torch

def sample_data(loader:DataLoader):
    loader_iter=iter(loader)
    while True:
        try:
            yield next(loader_iter)
        except StopIteration:
            loader_iter=iter(loader)
            yield next(loader_iter)

def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))