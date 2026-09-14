from DiffusionBinaryToMP.TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 100000,
        "batch_size": 3,
        "T": 3,
        "mode": 2, # 1 for channel 1, 2 for all channels
        "channel": 16,
        "channel_mult": [1, 2, 2],
        "attn": [2],
        "num_res_blocks": 1,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        # "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "device": "cpu", ### MAKE SURE YOU HAVE A GPU !!!
        "w": 1.8,
        "training_load_weight": None,
        "save_weight_dir": "./testc/",
        "test_load_weight": "ckpt_0_.pt",
        "sampled_dir": "./test/",
        "sampledImgName": "simple_ddpm.npy",
        "num_samples": 2,
        "nrow": 8,
        "save_interval": 1,
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
