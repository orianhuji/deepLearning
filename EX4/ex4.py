from argparse import ArgumentParser
import logging

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim

from src.blocks import UNet
from src.score_matching_rand import ScoreMatchingModel, ScoreMatchingModelConfig

import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torchvision import datasets
from torchvision.transforms import Resize, InterpolationMode
import ssl

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=1, type=int, choices=(0, 1))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    ssl._create_default_https_context = ssl._create_unverified_context

    import torch.utils.data as data_utils

    # Select training_set and testing_set
    transform =  transforms.Compose([transforms.Resize(32), transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    test_loader = datasets.MNIST("data", 
                                  train= False,
                                 download=True,
                                 transform=transform)

    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=10000, shuffle=True, num_workers=0)

    x = next(iter(test_loader))[0]
    x = x.view(-1,32*32).numpy()

    nn_module = UNet(1, 128, (1, 2, 4, 8))
    model = ScoreMatchingModel(
        nn_module=nn_module,
        input_shape=(1, 32, 32,),
        config=ScoreMatchingModelConfig(
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
        ),
    )
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    model.load_state_dict(torch.load("./EX4/ckpts/mnist_trained.pt",map_location=torch.device(args.device)))

    model.eval()

    input_sd = 127
    input_mean = 127
    x_vis = x[:32] * input_sd + input_mean

    # Degraded images
    x_true = x[:32].reshape(32, 1, 32, 32).copy()

    # Downscale and upscale functions
    def downscale(image, factor):
        # Convert numpy image to tensor
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension
        downscale_transform = Resize((32 // factor, 32 // factor), interpolation=InterpolationMode.BILINEAR)
        downscaled_image = downscale_transform(image)
        return downscaled_image.squeeze(0).numpy()  # Remove batch dimension

    def upscale(image, factor):
        # Convert numpy image to tensor
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension
        upscale_transform = Resize((32, 32), interpolation=InterpolationMode.BILINEAR)
        upscaled_image = upscale_transform(image)
        return upscaled_image.squeeze(0).numpy()  # Remove batch dimension

    # Task 1: Upscaling
    downscale_factor = 2
    x_downscaled = np.array([downscale(im, downscale_factor) for im in x_true])
    x_upscaled = np.array([upscale(im, downscale_factor) for im in x_downscaled])

    # Task 2: Inpainting
    mask_quarter = np.ones_like(x_true)
    mask_quarter[:, :, :16, :16] = 0  # Missing top-left quarter
    x_inpainted_quarter = x_true * mask_quarter

    mask_half = np.ones_like(x_true)
    mask_half[:, :, :16, :] = 0  # Missing top half
    x_inpainted_half = x_true * mask_half

    def compute_noise_value(sigma_noise, sigma_data):
        return sigma_noise / np.sqrt(sigma_noise**2 + sigma_data**2)

    # Sample generation
    for degraded_images, noise, description in [
        (x_upscaled, compute_noise_value((0.5)**0.5,1),"Upscaled"),
        (x_inpainted_quarter, compute_noise_value((0.25)**0.5, 1.0), "Inpainted Quarter"),
        (x_inpainted_half, compute_noise_value((0.5)**0.5,1), "Inpainted Half")
        ]:
        
        samples = model.sample(bsz=32, noise=1, x0=degraded_images, device=args.device).cpu().numpy()
        samples = rearrange(samples, "t b () h w -> t b (h w)")
        samples = samples * input_sd + input_mean

        nrows, ncols = 10, 3
        percents = min(len(samples),4)

        raster = np.zeros((nrows * 32, ncols * 32 * (percents + 2)), dtype=np.float32)

        degraded_images = degraded_images * input_sd + input_mean        

        # blocks of resulting images. Last row is the degraded image, before last row: the noise-free images. 
        # First rows show the denoising progression
        for percent_idx in range(percents):
            itr_num = int(round(percent_idx / (percents-1) * (len(samples)-1)))
            print(itr_num)
            for i in range(nrows * ncols):
                row, col = i // ncols, i % ncols
                offset = 32 * ncols * (percent_idx)
                raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = samples[itr_num][i].reshape(32, 32)

        # last block of nrow,ncol of input images
        for i in range(nrows * ncols):
            offset = 32 * ncols * percents
            row, col = i // ncols, i % ncols
            raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = x_vis[i].reshape(32, 32)

        for i in range(nrows * ncols):
            offset =  32 * ncols * (percents+1)
            row, col = i // ncols, i % ncols
            raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = degraded_images[i].reshape(32, 32)

        raster[:,::32*3] = 64

        plt.imsave(f"./EX4/examples/ex_mnist_{description}.png", raster, vmin=0, vmax=255, cmap='gray')
