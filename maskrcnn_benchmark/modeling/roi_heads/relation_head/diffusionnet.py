


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.random
import tqdm
import torch.fft as fft
import math
from matplotlib import pyplot as plt
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

from sklearn.cluster import MiniBatchKMeans, KMeans



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
def norm_layer(channels):
    return nn.GroupNorm(32, channels)

class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Linear(channels, channels)   #线性插值

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Linear(channels, channels//2)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, H= x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B, -1).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(H))
        attn = torch.mm(q * scale, k.T * scale)    #就是矩阵乘法
        attn = attn.softmax(dim=-1)
        h = torch.mm(attn, v)
        h = h.reshape(B, -1)
        h = self.proj(h)
        return h + x
#两个卷积层和shortcut（维持输入输出维度一致以及残差链接）
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.l1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels)
        )
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.l2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_channels, out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        
        h = self.l1(x)
        # Add time step embeddings
        h += self.time_emb(t)
        h = self.l2(h)
        return h + self.shortcut(x)

feat_dim=1024

class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=feat_dim,
        model_channels=256,
        out_channels=feat_dim,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),   #x∗σ(x)
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Linear(in_channels, model_channels))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                ch //= 2
                down_block_chans.append(ch)
                ds *= 2
                
        
        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x.device))
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)




def linear_beta_schedule(timesteps):
    scale = 20 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
def cosine_beta_schedule(timesteps):
    pass
class GaussianDiffusions(nn.Module):
    def __init__(
        self,
        timesteps=20,
        beta_schedule='linear'
    ):
        super(GaussianDiffusions, self).__init__()
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
        self.timesteps = timesteps

        self.timesteps = timesteps
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)     #α-bar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)   #根号α-bar
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)   #指前一个α？
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))

        self.model = UNetModel()

    def forward(self,feat):
        batch_size = feat.shape[0]
        timesteps = 20
        t = torch.randint(0, timesteps, (batch_size,)).long()
        loss,x_noisy = self.train_p(self.model, feat, t)
        return loss,x_noisy
    def train_p(self, model, x_start, t):
        # generate random noise
        noise = torch.randn_like(x_start,requires_grad=False)
        # get x_t
        x_noisy = self.forward_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = torch.cosine_similarity(noise, predicted_noise)
        loss = (1-torch.abs(loss)).mean()
        return loss, x_noisy
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        #print(t)  #bz个随机数，每个代表该image的最终t为多少
        out = a.to(t.device).gather(0, t).float()     
        #print(out)  #在总范围a（1000个），按t取出该image的系数;
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out     
    
    def forward_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)  #另一种用log的方式
        return sqrt_alphas_cumprod_t.to(x_start.device) * x_start + sqrt_one_minus_alphas_cumprod_t.to(x_start.device) * noise.to(x_start.device)
        #目前来说，t决定原图占比
    
    @torch.no_grad()
    def backward_sample_loop(self, x_t, shape):
        batch_size = shape[0]
        device = next(self.model.parameters()).device
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=device)
        img = x_t
        imgs = []
        for i in torch.range(20-1,0,-1):
            img = self.backward_sample(self.model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())

        a = torch.tensor(imgs)
        fake = a[-3:].reshape(3,batch_size,-1).sum(0)
        return fake   #n*1024
    @torch.no_grad()
    def backward_sample(self, model, x_t, t, clip_denoised=True):
        model_mean, model_variance, model_log_variance = self.backward_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # pred_img = model_mean + nonzero_mask * (0.5 * model_variance).exp() * noise     #加纯噪声  改进先不用log的把,这里得用根号
        pred_img = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return pred_img
    
    def backward_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:   #改进：需要截断吗
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    def posterior_mean_variance(self, x_start, x_t, t):   #求均值和方差
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

        


if __name__ == "__main__":

    

    feat = torch.rand(32,feat_dim)
    batch_size = feat.shape[0]
    timesteps = 1000
    gaussian_diffusion = GaussianDiffusion(timesteps)
    model = UNetModel()
    t = torch.randint(0, timesteps, (batch_size,)).long()
    loss = gaussian_diffusion.train(model, feat, t)
    print(loss)

    a = gaussian_diffusion.backward_sample_loop(model, feat.shape)
    a = torch.tensor(a)
    fake = a[-3:].reshape(3,32,-1).sum(0)
    # print(fake.shape)
    # print(len(a),a[-3:].reshape(3,4,-1))  #a[0]为第一个append进去的,a[-3:][0][0][0]到单个数，a[0][0][0]到单个数















