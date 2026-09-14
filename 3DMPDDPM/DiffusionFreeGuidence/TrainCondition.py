
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from DiffusionFreeGuidence .image_datasets import load_data
from DiffusionFreeGuidence import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition import UNet
from Scheduler import GradualWarmupScheduler
import pandas as pd


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
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0, labels).sum() / 1000.
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
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        # load the conditions_file csv file
        conditions = pd.read_csv(modelConfig["conditions_file"])
        # for each row in the conditions file
        phase_fractions = []
        for index, row in conditions.iterrows():
            for _ in range(1):
                # convert to tensor
                label = torch.tensor(row.values).to(device)
                label = torch.unsqueeze(label, 0)
                sampler = GaussianDiffusionSampler(
                    model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
                # Sampled from standard normal distribution
                noisyImage = torch.randn(
                    size=[1, 2, 64, 64, 64], device=device)
                sampledImgs = sampler(noisyImage, label)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                # calculate phase fraction
                p_0 = torch.mean(sampledImgs[:, 0, :, :, :])
                p_1 = torch.mean(sampledImgs[:, 1, :, :, :])
                name = f'porosity_{str(index/100)}_{str(_)}'
                # add the labels and phase fractions to the list
                print(row.values[0], row.values[1], p_0.item(), p_1.item())
                phase_fractions.append([row.values[0], row.values[1], p_0.item(), p_1.item()])
                if not os.path.exists(modelConfig["sampled_dir"]):
                    os.makedirs(modelConfig["sampled_dir"])
                sampledImgs = sampledImgs.cpu().numpy()
                np.save(os.path.join(modelConfig["sampled_dir"], name), sampledImgs)
            #     break
            # break
        # save the phase fractions to a csv file
        phase_fractions = pd.DataFrame(phase_fractions )
        phase_fractions.to_csv('./phase_fractions.csv', index=False)