"Largely taken and adapted from https://github.com/FirasGit/medicaldiffusion"

import numpy as np
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.optim    as optim

from pathlib import Path
from torch.optim import Adam, SGD
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from ddpm.lpips import LPIPS1

from tqdm import tqdm
from einops import rearrange
from util.lpips import PerceptualLoss
from torch.utils.data import DataLoader
from vq_gan_3d.model.vqgan import VQGAN

import matplotlib.pyplot as plt
from xraysyn.networks.drr_projector_new import DRRProjector
from ddpm.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity, LPIPS, RMSE, NRMSE
from ddpm.visualizer import tensor_back_to_unnormalization, tensor_back_to_unMinMax
import wandb

# helpers functions

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor  # 2
    remainder = num % divisor  # 0
    arr = [divisor] * groups  # [32]
    if remainder > 0:
        arr.append(remainder)
    return arr


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias


# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = (1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]))
    return torch.clip(betas, 0, 0.9999)


class FineStage(nn.Module):
    def __init__(
        self,
        fit_model,
        fine_model,
        *,
        image_size,
        input_img_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        pose_timesteps=3,
        loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
        vqgan_ckpt_bp=None,
        vqgan_ckpt_bp0=None,
        vqgan_ckpt_bp025=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.input_size = input_img_size
        self.num_frames = num_frames
        self.fit_model = fit_model
        self.fine_model = fine_model
        self.norm = NormLayer()
        self.perceptual_model = LPIPS1().eval()
        # self.x_encoder = x_encoder(n_class=1)

        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
            # for param in self.vqgan.parameters():
            #     param.requires_grad = False
        else:
            self.vqgan = None
        if vqgan_ckpt_bp:
            self.vqgan_bp = VQGAN.load_from_checkpoint(vqgan_ckpt_bp).cuda()
            self.vqgan_bp.eval()
        else:
            self.vqgan_bp = None

        if vqgan_ckpt_bp0:
            self.vqgan_bp0 = VQGAN.load_from_checkpoint(vqgan_ckpt_bp0).cuda()
            self.vqgan_bp0.eval()
        else:
            self.vqgan_bp0 = None

        if vqgan_ckpt_bp0:
            self.vqgan_bp025 = VQGAN.load_from_checkpoint(vqgan_ckpt_bp025).cuda()
            self.vqgan_bp025.eval()
        else:
            self.vqgan_bp025 = None



        betas = cosine_beta_schedule(timesteps)
        betas_square = torch.square(betas)
        betas_cumprod = torch.cumprod(betas, dim=0)
        betas_cumprod_prev = F.pad(betas_cumprod[:-1], (1, 0), value=1.)
        betas_cumprod_square = torch.square(betas_cumprod)
        betas_cumprod_prev_square = torch.square(betas_cumprod_prev)


        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.pose_timesteps = int(pose_timesteps)
        self.loss_type = loss_type
        self.fp = DRRProjector(
            mode="forward", volume_shape=(self.input_size, self.input_size, self.input_size), detector_shape=(self.input_size, self.input_size),
            pixel_size=(1.0, 1.0), interp="trilinear", source_to_detector_distance=1200).to("cuda")
        self.bp = DRRProjector(
            mode="backward", volume_shape=(self.input_size, self.input_size, self.input_size), detector_shape=(self.input_size, self.input_size),
            pixel_size=(1.0, 1.0), interp="trilinear", source_to_detector_distance=1200).to("cuda")

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (x_t - extract(self.betas_cumprod, t, x_t.shape) * noise)


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, ct_bp_xT=None, ct_bp_0=None, ct_bp_025=None, ct_bp_loop=None, ct_bp=None, ct_vq=None, self_cond=None, cond=None, cond_scale=1.):
        # x_recon = self.predict_start_from_noise(
        #     x, t=t, noise=self.denoise_fn.forward_with_cond_scale(torch.cat((ct_bp, x), dim=1), t, self_cond=self_cond, cond=cond, cond_scale=cond_scale))

        ct_pred = self.fine_model(torch.cat((ct_bp_xT, ct_bp_0, ct_bp_025, x), dim=1), t)    # pred ct
        x_recon = ct_pred    # 估计的x0

        # x_recon = self.denoise_fn(x, t)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, ct_bp_xT=None, ct_bp_0=None,ct_bp_025=None, ct_vq=None, self_cond=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, self_cond=self_cond, clip_denoised=clip_denoised, ct_bp_xT=ct_bp_xT, ct_bp_0=ct_bp_0, ct_bp_025=ct_bp_025, ct_vq=ct_vq, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, ct_bp_xT=None, ct_bp_0=None, ct_bp_025=None, ct_vq=None, self_cond=None, cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long), ct_bp_xT=ct_bp_xT, ct_bp_0=ct_bp_0, ct_bp_025=ct_bp_025, ct_vq=ct_vq, self_cond=self_cond, cond=cond, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, ct, cond=None, cond_scale=1., batch_size=8):

        device = next(self.fit_model.parameters()).device
        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        t = torch.tensor([self.pose_timesteps-1], device='cuda').long()

        xray = self.q_sample_pose_xT(ct, t, self.fp)   # xT
        xray_0 = self.q_sample_pose_0(ct, t, self.fp)
        ct_bp_xT = self.xray_bp(xray, torch.tensor([self.pose_timesteps-1]), self.bp)
        ct_bp_0 = self.xray_bp_0(xray_0, t, self.bp)
        xray_025 = self.q_sample_pose_025(ct, t, self.fp)
        ct_bp_025 = self.xray_bp_025(xray_025, t, self.bp)

        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                ct_bp_xT = self.vqgan_bp.encode(ct_bp_xT, quantize=False, include_embeddings=True)
                ct_bp_0 = self.vqgan_bp0.encode(ct_bp_0, quantize=False, include_embeddings=True)
                # ct_bp_025 = self.vqgan_bp.encode(ct_bp_025, quantize=False, include_embeddings=True)
                # ct_zeros = self.vqgan_bp.encode(ct_zeros, quantize=False, include_embeddings=True)
                # ct_bp_xT = ((ct_bp_xT - self.vqgan_bp.codebook.embeddings.min()) /
                #          (self.vqgan_bp.codebook.embeddings.max() - self.vqgan_bp.codebook.embeddings.min())) * 2.0 - 1.0
                # ct_bp_0 = ((ct_bp_0 - self.vqgan_bp0.codebook.embeddings.min()) /
                #             (self.vqgan_bp0.codebook.embeddings.max() - self.vqgan_bp0.codebook.embeddings.min())) * 2.0 - 1.0
                # ct_bp_025 = ((ct_bp_025 - self.vqgan_bp.codebook.embeddings.min()) /
                #             (self.vqgan_bp.codebook.embeddings.max() - self.vqgan_bp.codebook.embeddings.min())) * 2.0 - 1.0

                ct_pred_coarse = self.fit_model(torch.cat((ct_bp_xT, ct_bp_0), dim=1), t)
                ct_pred_coarse = (((ct_pred_coarse + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                      self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()
                ct_pred_coarse_wovq = self.vqgan.decode(ct_pred_coarse, quantize=True)
                xray_pred = self.q_sample_pose_025(ct_pred_coarse_wovq, torch.tensor([0]), self.fp)
                ct_bp_xray_pred = self.xray_bp_025(xray_pred, t, self.bp)
                ct_bp_xray_pred = self.vqgan_bp025.encode(ct_bp_xray_pred, quantize=False, include_embeddings=True)
                # ct_bp_xray_pred = ((ct_bp_xray_pred - self.vqgan_bp025.codebook.embeddings.min()) /
                #                    (self.vqgan_bp025.codebook.embeddings.max() - self.vqgan_bp025.codebook.embeddings.min())) * 2.0 - 1.0

                # normalize to -1 and 1
                # ct_bp_xT = ((ct_bp_xT - torch.min(ct_bp_xT)) / (torch.max(ct_bp_xT) - torch.min(ct_bp_xT))) * 2.0 - 1.0
                # ct_bp_xT = ((ct_bp_xT - self.vqgan_bp.codebook.embeddings.min()) /
                #          (self.vqgan_bp.codebook.embeddings.max() - self.vqgan_bp.codebook.embeddings.min())) * 2.0 - 1.0
            # ct_pred = self.fit_model(torch.cat((ct_bp_xT, ct_bp_0), dim=1), torch.full((1,), self.num_timesteps - 1, device=device, dtype=torch.long))
            ct_pred = self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), ct_bp_xT=ct_bp_xT, ct_bp_0=ct_bp_0, ct_bp_025=ct_bp_xray_pred,
                                         cond=cond, cond_scale=cond_scale)
            if isinstance(self.vqgan, VQGAN):
                # denormalize TODO: Remove eventually
                ct_pred = (((ct_pred + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                      self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

                ct_pred = self.vqgan.decode(ct_pred, quantize=True)

            ct_pred = torch.clip(ct_pred, 0, 1)
            ct_pred = torch.where(ct_pred < 0.04, 0, ct_pred)
            ct_pred = torch.where(ct_pred > 0.87, 1, ct_pred)
            xray_pred = self.q_sample_pose_025(ct_pred, torch.tensor([0]), self.fp)



        x_ray_I_0 = self.q_sample_pose_025(ct, t, self.fp)




        return [ct_pred, x_ray_I_0, xray_pred]
    # x_ray_I_pred, xray, x_ray_I_0_pred, x_ray_I_0, x_ray_I_005_pred, x_ray_I_005,


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod,
                        t, x_start.shape) * noise
        )

    def q_sample1(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return 0.5*(x_start + extract(self.betas_cumprod, t, x_start.shape) * noise)



    def xray_bp_loop(self,xray_i_list, t, bp):
        num = 0
        ct_i_lists = []
        ct_zeros = torch.zeros((1, 1, self.input_size, self.input_size, self.input_size)).cuda()
        for t_item in t:
            iff = t_item.item() + 1 - (self.pose_timesteps - 1)
            m = 0
            ct_i_list = 0
            if iff < 0:
                for i in range(t_item+1, self.pose_timesteps-1):
                    view = -0.5*(i/(self.pose_timesteps-1)) # i=1时为右侧面
                    pose = [view, 0, 0, 0, 0, 0]
                    T_in = get_T(pose)
                    ct_i = bp(xray_i_list[num][m].unsqueeze(dim=0), T_in)
                    m = m + 1
                    ct_i_list = ct_i_list + ct_i
                ct_i_lists.append(ct_i_list)
            else:
                ct_i_lists.append(ct_zeros)
            num = num + 1
        ct_i_lists = torch.cat(ct_i_lists)
        return ct_i_lists

    def q_sample_pose_xT(self, ct, t, fp):
        b, _, _, _, _ = ct.shape
        view_t = -0.5 * (t / (self.pose_timesteps - 1)) if self.pose_timesteps > 1 else torch.randint(
            self.pose_timesteps - 1, self.pose_timesteps, (b,), device='cuda').long()
        xray_list = []
        i = 0
        for view in view_t:
            pose = [-0.5, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            x_ray = fp(ct[i].unsqueeze(dim=0), T_in)
            i = i + 1
            # x_ray = self.norm(x_ray)
            xray_list.append(x_ray)
        xray_list = torch.cat(xray_list)
        # xray_list1 = (xray_list+1)/2
        # tensor2img = T.ToPILImage()
        # im = tensor2img(xray_list1[0])
        # im.show()

        return xray_list





    def xray_bp(self, xray, t, bp):
        b, _, _, _ = xray.shape
        view_t = -0.5*(t/(self.pose_timesteps-1)) if self.pose_timesteps > 1 else torch.randint(self.pose_timesteps-1, self.pose_timesteps, (b,), device='cuda').long()
        ct_list = []
        i = 0
        for view in view_t:
            pose = [-0.5, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            ct_bp = bp(xray[i].unsqueeze(dim=0), T_in)
            # ct_bp = self.norm(ct_bp)
            i = i + 1
            ct_list.append(ct_bp)
        ct_list = torch.cat(ct_list)
        # xray_pred = self.fp(ct_list, T_in)
        # xray_pred = torch.squeeze(xray_pred).to("cpu")
        # xray_pred = (xray_pred - torch.min(xray_pred)) / (torch.max(xray_pred) - torch.min(xray_pred))
        # print(torch.max(xray_pred))
        # print(torch.min(xray_pred))
        # tensor2img = T.ToPILImage()
        # im2 = tensor2img(xray_pred)
        # im2.show()

        return ct_list

    def xray_bp_0(self, xray, t, bp):
        b, _, _, _ = xray.shape
        view_t = -0.5*(t/(self.pose_timesteps-1)) if self.pose_timesteps > 1 else torch.randint(self.pose_timesteps-1, self.pose_timesteps, (b,), device='cuda').long()
        ct_list = []
        i = 0
        for view in view_t:
            pose = [0, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            ct_bp = bp(xray[i].unsqueeze(dim=0), T_in)
            # ct_bp = self.norm(ct_bp)
            i = i + 1
            ct_list.append(ct_bp)
        ct_list = torch.cat(ct_list)
        # xray_pred = self.fp(ct_list, T_in)
        # xray_pred = torch.squeeze(xray_pred).to("cpu")
        # xray_pred = (xray_pred - torch.min(xray_pred)) / (torch.max(xray_pred) - torch.min(xray_pred))
        # print(torch.max(xray_pred))
        # print(torch.min(xray_pred))
        # tensor2img = T.ToPILImage()
        # im2 = tensor2img(xray_pred)
        # im2.show()

        return ct_list

    def xray_bp_025(self, xray, t, bp):
        b, _, _, _ = xray.shape
        view_t = -0.5*(t/(self.pose_timesteps-1)) if self.pose_timesteps > 1 else torch.randint(self.pose_timesteps-1, self.pose_timesteps, (b,), device='cuda').long()
        ct_list = []
        i = 0
        for view in view_t:
            pose = [-0.25, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            ct_bp = bp(xray[i].unsqueeze(dim=0), T_in)
            # ct_bp = self.norm(ct_bp)
            i = i + 1
            ct_list.append(ct_bp)
        ct_list = torch.cat(ct_list)
        # xray_pred = self.fp(ct_list, T_in)
        # xray_pred = torch.squeeze(xray_pred).to("cpu")
        # xray_pred = (xray_pred - torch.min(xray_pred)) / (torch.max(xray_pred) - torch.min(xray_pred))
        # print(torch.max(xray_pred))
        # print(torch.min(xray_pred))
        # tensor2img = T.ToPILImage()
        # im2 = tensor2img(xray_pred)
        # im2.show()

        return ct_list


    def q_sample_pose_005(self, ct, t, fp):
        view_t = -0.5*(t/(self.pose_timesteps-1))
        xray_list = []
        i = 0
        for view in view_t:
            pose = [0, 0.5, 0, 0, 0, 0]
            T_in = get_T(pose)
            x_ray = fp(ct[i].unsqueeze(dim=0), T_in)
            i = i + 1
            x_ray = self.norm(x_ray)
            xray_list.append(x_ray)
        xray_list = torch.cat(xray_list)
        # tensor2img = T.ToPILImage()
        # im = tensor2img(xray_list[0])
        # im.show()

        return xray_list

    def q_sample_pose_0(self, ct, t, fp, norm=True):
        view_t = -0.5*(t/(self.pose_timesteps-1))
        xray_list = []
        i = 0
        for view in view_t:
            pose = [0, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            x_ray = fp(ct[i].unsqueeze(dim=0), T_in)
            i = i + 1
            if norm:
                x_ray = self.norm(x_ray)
            xray_list.append(x_ray)
        xray_list = torch.cat(xray_list)
        # tensor2img = T.ToPILImage()
        # im = tensor2img(xray_list[0])
        # im.show()

        return xray_list

    def q_sample_pose_025(self, ct, t, fp, norm=True):
        view_t = -0.5*(t/(self.pose_timesteps-1))
        xray_list = []
        i = 0
        for view in view_t:
            pose = [-0.25, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            x_ray = fp(ct[i].unsqueeze(dim=0), T_in)
            i = i + 1
            if norm:
                x_ray = self.norm(x_ray)
            xray_list.append(x_ray)
        xray_list = torch.cat(xray_list)
        # tensor2img = T.ToPILImage()
        # im = tensor2img(xray_list[0])
        # im.show()

        return xray_list

    def q_sample_pose(self, ct, t, fp):
        b, _, _, _, _ = ct.shape
        view_t = -0.5*(t/(self.pose_timesteps-1)) if self.pose_timesteps > 1 else torch.randint(self.pose_timesteps-1, self.pose_timesteps, (b,), device='cuda').long()
        xray_list = []
        i = 0
        for view in view_t:
            pose = [-0.5, 0, 0, 0, 0, 0]
            T_in = get_T(pose)
            x_ray = fp(ct[i].unsqueeze(dim=0), T_in)
            i = i + 1
            x_ray = self.norm(x_ray)
            xray_list.append(x_ray)
        xray_list = torch.cat(xray_list)
        # tensor2img = T.ToPILImage()
        # im = tensor2img(xray_list[0])
        # im.show()

        return xray_list


    def p_losses(self, ct_vq, ct_bp_xT, ct_bp_0, ct_bp_025, ct, t, noise=None, **kwargs):

        b, c, f, h, w, device = *ct_vq.shape, ct_vq.device
        noise = default(noise, lambda: torch.randn_like(ct_vq))
        x_noisy = self.q_sample(x_start=ct_vq, t=t, noise=noise)
        # print(ct_bp_xT.max())
        # print(ct_bp_xT.min())
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                ct_bp_xT = self.vqgan_bp.encode(ct_bp_xT, quantize=False, include_embeddings=True)
                ct_bp_0 = self.vqgan_bp0.encode(ct_bp_0, quantize=False, include_embeddings=True)
                # ct_bp_025 = self.vqgan_bp.encode(ct_bp_025, quantize=False, include_embeddings=True)
                # ct_bp_xT = ((ct_bp_xT - self.vqgan_bp.codebook.embeddings.min()) /
                #             (self.vqgan_bp.codebook.embeddings.max() - self.vqgan_bp.codebook.embeddings.min())) * 2.0-1.0
                # ct_bp_0 = ((ct_bp_0 - self.vqgan_bp0.codebook.embeddings.min()) /
                #             (self.vqgan_bp0.codebook.embeddings.max() - self.vqgan_bp0.codebook.embeddings.min())) * 2.0-1.0
                # ct_bp_025 = ((ct_bp_025 - self.vqgan_bp.codebook.embeddings.min()) /
                #            (self.vqgan_bp.codebook.embeddings.max() - self.vqgan_bp.codebook.embeddings.min())) * 2.0 - 1.0
                ct_pred_coarse = self.fit_model(torch.cat((ct_bp_xT, ct_bp_0), dim=1), t, **kwargs)
                ct_pred_coarse = (((ct_pred_coarse + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                                    self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()
                ct_pred_coarse_wovq = self.vqgan.decode(ct_pred_coarse, quantize=True)
                xray_pred = self.q_sample_pose_025(ct_pred_coarse_wovq, t, self.fp)
                ct_bp_xray_pred = self.xray_bp_025(xray_pred, t, self.bp)
                ct_bp_xray_pred = self.vqgan_bp025.encode(ct_bp_xray_pred, quantize=False, include_embeddings=True)
                # ct_bp_xray_pred = ((ct_bp_xray_pred - self.vqgan_bp025.codebook.embeddings.min()) /
                #              (self.vqgan_bp025.codebook.embeddings.max() - self.vqgan_bp025.codebook.embeddings.min())) * 2.0 - 1.0
        ct_pred = self.fine_model(torch.cat((ct_bp_xT, ct_bp_0, ct_bp_xray_pred, x_noisy), dim=1), t, **kwargs)


        if self.loss_type == 'l1':
            loss4 = F.l1_loss(ct_vq, ct_pred)
            # loss2 = loss + loss1
        elif self.loss_type == 'l2':
            # loss2 = F.mse_loss(ct, ct_pred)
            # loss3 = F.mse_loss(noise, x_recon)
            # loss4 = F.l1_loss(xray_fine, xray_gt)
            # loss4 = F.mse_loss(ct_vq, x_recon) + 1/3 * (F.l1_loss(xray_gt, xray_pred) + F.l1_loss(xray_gt005, xray_pred005) + F.l1_loss(xray_gt5, xray_pred5))
            loss4 = F.mse_loss(ct_vq, ct_pred)
            # loss5 = loss3 + loss4
        else:
            raise NotImplementedError()

        return loss4

    def forward(self, ct, *args, **kwargs):
        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                ct_vq = self.vqgan.encode(ct, quantize=False, include_embeddings=True)
                # print(ct_vq.max())
                # print(ct_vq.min())
                # normalize to -1 and 1
                ct_vq = ((ct_vq - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
                # print(torch.max(ct_vq))
                # print(torch.min(ct_vq))
        b, device, img_size, = ct.shape[0], ct.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        T = torch.randint(self.pose_timesteps-1, self.pose_timesteps, (b,), device=device).long()


        xrayT = self.q_sample_pose_xT(ct, t, self.fp)
        xray_0 = self.q_sample_pose_0(ct, t, self.fp)
        xray_025 = self.q_sample_pose_025(ct, t, self.fp)
        ct_bp_xT = self.xray_bp(xrayT, T, self.bp)
        ct_bp_0 = self.xray_bp_0(xray_0, T, self.bp)
        ct_bp_025 = self.xray_bp_025(xray_025, T, self.bp)

        return self.p_losses(ct_vq=ct_vq, ct_bp_xT=ct_bp_xT, ct_bp_0=ct_bp_0, ct_bp_025=ct_bp_025, ct=ct, t=t, *args, **kwargs)

class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, inp):
        # print(inp.shape)
        inp = inp - inp.min()
        return inp/(inp.max()-inp.min())

# trainer class


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    # print("tensor_max", torch.max(tensor))
    # print("tensor_min", torch.min(tensor))
    # tensor = (tensor + 1) / 2
    tensor = ((tensor - tensor.min()) / (tensor.max()-tensor.min())) * 1.0
    # print("tensor1_max", torch.max(tensor))
    # print("tensor1_min", torch.min(tensor))
    # tensor = (tensor - torch.min(tensor)) * 2 / (torch.max(tensor) - torch.min(tensor)) - 1
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

def get_T(inp):
    param =np.asarray(inp)
    param = param * np.pi
    T = get_6dofs_transformation_matrix(param[3:], param[:3])
    T = torch.FloatTensor(T[np.newaxis, ...]).to("cuda")
    return torch.cat([T, T, T, T])

def get_6dofs_transformation_matrix(u, v):
    """ https://arxiv.org/pdf/1611.10336.pdf
    """
    x, y, z = u
    theta_x, theta_y, theta_z = v

    # rotate theta_z
    rotate_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # rotate theta_y
    rotate_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    # rotate theta_x and translate x, y, z
    rotate_x_translate_xyz = np.array([
        [1, 0, 0, x],
        [0, np.cos(theta_x), -np.sin(theta_x), y],
        [0, np.sin(theta_x), np.cos(theta_x), z],
        [0, 0, 0, 1]
    ])

    return rotate_x_translate_xyz.dot(rotate_y).dot(rotate_z)

# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        folder=None,
        dataset=None,
        val_dataset=None,
        visual_dataset=None,
        cond_dataset=None,
        *,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        test_batch_size=16,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        val_every = 20,
        results_folder='./results',
        num_sample_rows=10,
        max_grad_norm=None,
        num_workers=20,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.val_every = val_every

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.cfg = cfg
        self.ds = dataset
        self.val_ds = val_dataset
        self.visual_ds = visual_dataset
        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)
        self.val_dl = DataLoader(self.val_ds, batch_size=test_batch_size,
                        shuffle=False, pin_memory=True, num_workers=num_workers)
        self.visual_dl = DataLoader(self.visual_ds, batch_size=test_batch_size,
                        shuffle=False, pin_memory=True, num_workers=0)

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        wandb.init(project="twoview_diffusion")
        artifact = wandb.Artifact(name='twoview_diffusion', type='project')
        artifact.add_file('E:/workspace/AX2CT/twoview_diffusion/ddpm/diffusion.py')

        wandb.watch(self.model, log="all", log_freq=100, log_graph=False)

        print(print('#Test images = %d' % len(self.val_dl)))
        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(
            self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        # for name, parameter in diffusion_model.named_parameters():
        #     if "vqgan" in name:
        #         parameter.requires_grad = False
        # vqgan_params = []
        # params = []
        # for name, parameter in diffusion_model.named_parameters():
        #     if 'vqgan' in name:
        #         vqgan_params.append(parameter)
        #     else:
        #         params.append(parameter)
        # print(vqgan_params)
        self.opt = Adam(filter(lambda p: p.requires_grad, diffusion_model.parameters()), lr=train_lr)
        # self.opt1 = SGD(filter(lambda p: p.requires_grad, diffusion_model.parameters()), lr=0.1)
        # self.opt1 = Adam(filter(lambda p: p.requires_grad, diffusion_model.vqgan.parameters()), lr=0)
        # self.opt = Adam([{'params': vqgan_params, 'lr': 0}, {'params': params, 'lr': train_lr}])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=40000, eta_min=1e-10)
        # self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()
        self.txt_path = self.results_folder / 'train_log.txt'

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)


        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def load_coarse(self, milestone, map_location=None, **kwargs):
        data = torch.load(milestone)
        data = data['model']
        model_params = self.model.state_dict()
        same_parsms = {k: v for k, v in data.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        self.model.load_state_dict(model_params, **kwargs)

    def train_log(self, path, txt):
        with open(path, "a") as f:
            f.write(txt)
            f.write("\n")

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)['data']
                data = data.cuda()

                with autocast(enabled=self.amp):
                    mse_loss = self.model(  # diffusion的forward
                        data,
                        self_cond=None,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )

                    loss = mse_loss
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
            if self.step != 0 and self.step % 10 == 0:
                print(f'{self.step}: loss_total: {loss.item()} loss_p:{mse_loss.item()}  loss_mse:{mse_loss.item()}')
                wandb.log({'loss': loss.item()}, step=self.step)
                # wandb.log({'p_loss': p_loss.item()}, step=self.step)
                wandb.log({'mse_loss': mse_loss.item()}, step=self.step)
            if self.step != 0 and self.step % 50 == 0:
                self.train_log(self.txt_path, f'{self.step}: {loss.item()}')


            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                # self.scaler.unscale_(self.opt1)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            # self.scaler.step(self.opt1)
            self.scaler.update()
            self.opt.zero_grad()
            # self.opt1.zero_grad()
            self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()


            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                print("lr", self.opt.state_dict()['param_groups'][0]['lr'])
                self.ema_model.eval()
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    num_samples = self.num_sample_rows ** 2
                    batches = num_to_groups(num_samples, self.batch_size)

                    out = list(map(lambda n: self.ema_model.sample(ct=self.val_ds[0]['data'].unsqueeze(dim=0).cuda(), batch_size=n), batches))
                    all_videos_list = out[0][0].unsqueeze(dim=0)
                    # xray_pred = out[0][1].squeeze(0)
                    # xray_list = out[0][2].squeeze(0)
                    xray0_pred = out[0][1].squeeze(0)
                    xray0 = out[0][2].squeeze(0)
                    # xray005_pred = out[0][5].squeeze(0)
                    # xray005 = out[0][6].squeeze(0)
                    all_videos_list = torch.cat(list(all_videos_list), dim=0)

                one_gif = rearrange(
                    list(all_videos_list), '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                videoGT_path = str(self.results_folder / str(f'{milestone}GT.gif'))
                video_tensor_to_gif(one_gif, video_path)
                video_tensor_to_gif(self.val_ds[0]['data'], videoGT_path)
                log = {**log, 'sample': video_path}
                # turn to x_ray
                tensor2img = T.ToPILImage()
                x_ray0_pred_image = tensor2img(xray0_pred)
                plt.imsave(self.results_folder / f'xray0-{milestone}.jpg', x_ray0_pred_image, cmap='gray')
                x_ray0_image = tensor2img(xray0)
                plt.imsave(self.results_folder / f'xray0GT-{milestone}.jpg', x_ray0_image, cmap='gray')

                # Selects one 2D image from each 3D Image
                B, C, D, H, W = all_videos_list.shape
                # frame_idx = torch.randint(0, D, [B]).cuda()
                frame_idx = torch.tensor([60]).cuda()
                frame_idx_selected = frame_idx.reshape(
                    -1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
                frames = torch.gather(all_videos_list, 2, frame_idx_selected).squeeze(2)
                plt.imsave(self.results_folder / f'ct_pred-{milestone}.jpg', frames.cpu().squeeze(), cmap='gray')

                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')

    def val(self):
        percept = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
        self.ema_model.eval()

        tbar = tqdm(self.val_dl)
        with torch.no_grad():
            avg_dict = dict()
            for i, data in enumerate(tbar):
                data = data['data'].cuda()
                out_ct, _, _ = self.ema_model.sample(ct=data, batch_size=self.test_batch_size)
                generate_CT = out_ct.data.clone().cpu().numpy().squeeze(1)
                generate_CT = np.transpose(generate_CT, (0, 2, 1, 3))
                generate_CT_TENSOR = torch.tensor(generate_CT)
                results_folder = Path('E:/workspace/AX2CT/newview_diffusion/results')
                video_path = str(results_folder / str(f'{i + 1}.gif'))
                video_tensor_to_gif(generate_CT_TENSOR, video_path)

                print(generate_CT.max())
                print(generate_CT.min())

                real_CT = data.data.clone().cpu().numpy().squeeze(1)
                real_CT = np.transpose(real_CT, (0, 2, 1, 3))
                # real_CT_TENSOR = torch.tensor(real_CT)
                # results_folder_GT = Path('E:/workspace/AX2CT/newview_diffusion/result_gt')
                # video_path1 = str(results_folder_GT / str(f'{i+1}.gif'))
                # video_tensor_to_gif(real_CT_TENSOR, video_path1)
                # clip generate_CT

                # CT range 0-1
                mae0 = MAE(real_CT, generate_CT, size_average=False)
                mse0 = MSE(real_CT, generate_CT, size_average=False)
                cosinesimilarity = Cosine_Similarity(real_CT, generate_CT, size_average=False)
                ssim = Structural_Similarity(real_CT, generate_CT, size_average=False, PIXEL_MAX=1.0)
                # ssim0 = Structural_Similarity(real_CT, generate_CT0, size_average=False, PIXEL_MAX=1.0)
                lpips = LPIPS(torch.from_numpy(real_CT).unsqueeze(0),
                              torch.from_numpy(generate_CT).unsqueeze(0), percept, size_average=False)
                # lpips0 = LPIPS(torch.from_numpy(real_CT).unsqueeze(0),
                #               torch.from_numpy(generate_CT0).unsqueeze(0), percept, size_average=False)
                # ssim1 = Structural_Similarity(real_CT, generate_CT1, size_average=False, PIXEL_MAX=1.0)
                # CT range 0-2500
                generate_CT_transpose = tensor_back_to_unMinMax(generate_CT, 0, 2500).astype(np.int32)
                # generate_CT_transpose0 = tensor_back_to_unMinMax(generate_CT0, 0, 2500).astype(np.int32)
                # generate_CT_transpose1 = tensor_back_to_unMinMax(generate_CT1, 0, 2500).astype(np.int32)
                real_CT_transpose = tensor_back_to_unMinMax(real_CT, 0, 2500).astype(np.int32)
                # psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=2500)
                psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False,
                                                 PIXEL_MAX=2500)
                # psnr0 = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose0, size_average=False,
                #                                  PIXEL_MAX=2500)
                # psnr1 = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose1, size_average=False, PIXEL_MAX=2500)
                mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
                # mae1 = MAE(real_CT_transpose, generate_CT_transpose1, size_average=False)
                mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
                rmse = RMSE(real_CT_transpose, generate_CT_transpose, size_average=False)
                nrmse = NRMSE(real_CT_transpose, generate_CT_transpose)
                # nrmse0 = NRMSE(real_CT_transpose, generate_CT_transpose0)

                metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MSE', mse), ('RMSE', rmse),
                                ('NRMSE', nrmse), ('NRMSE0', nrmse),
                                ('LPIPS-TS', lpips[0]), ('CosineSimilarity', cosinesimilarity), ('PSNR-1', psnr[0]),
                                ('LPIPS-CS', lpips[1]),
                                ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('LPIPS-SS', lpips[2]),
                                ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]),
                                ('PSNR-avg', psnr[3]), ('PSNR0-avg', psnr[3]), ('SSIM-avg', ssim[3]),
                                ('SSIM0-avg', ssim[3]),
                                ('LPIPS-avg', lpips[3]), ('LPIPS0-avg', lpips[3])]
                # metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MAE1', mae1), ('MSE', mse),
                #                 ('CosineSimilarity', cosinesimilarity),
                #                 ('psnr-3d', psnr_3d), ('PSNR-1', psnr[0]),
                #                 ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]), ('PSNR1-avg', psnr1[3]),
                #                 ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3]), ('SSIM1-avg', ssim1[3])]
                for key, value in metrics_list:
                    if avg_dict.get(key) is None:
                        avg_dict[key] = [] + value.tolist()
                    else:
                        avg_dict[key].extend(value.tolist())

            for key, value in avg_dict.items():
                print('### --{}-- total: {}; avg: {} '.format(key, len(value), np.round(np.mean(value), 7)))
                avg_dict[key] = np.mean(value)
                self.train_log(self.txt_path, '### --{}-- total: {}; avg: {} '.format(key, len(value),
                                                                                      np.round(np.mean(value), 7)))
        return avg_dict



if __name__ == '__main__':
    pass
