
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from DiffusionST_multi_sample_per_it.image_datasets import load_data
from DiffusionST_multi_sample_per_it import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionST_multi_sample_per_it.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    dataloader = load_data(data_dir='./datasets/illite', batch_size=modelConfig["batch_size"], class_cond=False, deterministic=False)
    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i in reversed(range(modelConfig["T"])):
                t = torch.ones(size=(modelConfig["batch_size"], ), device=modelConfig["device"], dtype=torch.int64) * i
                for images, labels in tqdmDataLoader:
                    # train
                    optimizer.zero_grad()
                    x_0 = images.to(device)
                    loss = trainer(x_0, t).sum() / 1000.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(), modelConfig["grad_clip"])
                    optimizer.step()
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    })
                    break
        warmUpScheduler.step()
        if e % modelConfig["save_interval"] == 0:
            torch.save(
                net_model.state_dict(),
                os.path.join(modelConfig["save_weight_dir"], f'ckpt_{str(e)}_.pt'),
            )

def eval(modelConfig: Dict):
    # load model and evaluate
    generated_samples = []
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        number_of_batches = modelConfig["num_samples"] // modelConfig["batch_size"]
        for _ in range(number_of_batches):
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 1, 64, 64, 64], device=device)
            sampledImgs = sampler(noisyImage)
            sampledImgs = sampledImgs * 0.5 + 0.5
            generated_samples.append(sampledImgs)
        sampledImgs = torch.cat(generated_samples, dim=0).cpu().numpy()
        print("sampledImgs shape: ", sampledImgs.shape)
        if not os.path.exists(modelConfig["sampled_dir"]):
            os.makedirs(modelConfig["sampled_dir"])
        np.save(os.path.join(modelConfig["sampled_dir"], modelConfig["sampledImgName"]), sampledImgs)