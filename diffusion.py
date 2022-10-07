import argparse
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm import tqdm

from dataset import *
from model import UNetModel
from utils import setup_logging, extract_into_tensor

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(
            self,
            noise_steps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            img_size=256,
            beta_scheduler='linear',
    ):
        self.num_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.beta_scheduler = beta_scheduler

        self.betas = self.get_named_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)
        self.log_one_minus_alpha_bars = np.log(1.0 - self.alpha_bars)
        self.sqrt_recip_alpha_bars = np.sqrt(1.0 / self.alpha_bars)
        self.sqrt_recipm1_alpha_bars = np.sqrt(1.0 / self.alpha_bars - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

    def get_named_beta_schedule(self):
        """
        Get a pre-defined beta schedule for the given name.
        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if self.beta_scheduler == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / self.num_steps
            self.beta_start *= scale
            self.beta_end *= scale
            return np.linspace(
                self.beta_start,
                self.beta_end,
                self.num_steps,
                dtype=np.float64
            )
        elif self.beta_scheduler == "cosine":
            return self.betas_from_alpha_bar(
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(
                f"unknown beta schedule: {self.beta_scheduler}")

    def betas_from_alpha_bar(self, alpha_bars, max_beta=0.999):
        betas = []
        for i in range(self.num_steps):
            t1 = i / self.num_steps
            t2 = (i + 1) / self.num_steps
            betas.append(min(1 - alpha_bars(t2) / alpha_bars(t1), max_beta))
        return np.array(betas)

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)

        return \
            extract_into_tensor(self.sqrt_alpha_bars, t, x_start.shape) * x_start \
            + extract_into_tensor(self.sqrt_one_minus_alpha_bars, t, x_start.shape) * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.num_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(device)
            for i in tqdm(reversed(range(1, self.num_steps)), position=0):
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t)

                alpha = self.alphas[t][:, None, None, None]
                alpha_bar = self.alpha_bars[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                predicted_mean = 1 / torch.sqrt(alpha) * \
                                 (x - beta / (torch.sqrt(1 - alpha_bar)) * predicted_noise)

                if i > 1:
                    x = predicted_mean + torch.sqrt(beta) * torch.randn_like(x)
                else:
                    x = predicted_mean

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args, dataloader):
    setup_logging(args.run_name)

    model = UNetModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.lr = 3e-4
    args.image_size = 256
    args.epochs = 500
    args.batch_size = 64
    args.run_name = "DDPM_Unconditional"
    dataloader = create_dataloader(args.batch_size, args.image_size)
    train(args, dataloader)

    # DiffusionModel = Diffusion(beta_scheduler='cosine')
    # image = next(iter(dataloader))[0]
    # plt.figure(figsize=(15, 15))
    # plt.axis('off')
    # num_images = 10
    # T = 1000
    # step_size = int(T / num_images)
    #
    # for idx in range(0, T, step_size):
    #     t = torch.Tensor([idx]).type(torch.int64)
    #     plt.subplot(1, num_images + 1, int((idx / step_size) + 1))
    #     image, noise = DiffusionModel.noise_image(image, t)
    #     show_tensor_image(image)
    # plt.savefig('testsubplots_cosine.jpg')