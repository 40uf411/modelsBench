from DiffusionMP.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 100001,
        "batch_size": 2,
        "T": 1000,
        "sampling_T": 50,
        "number_of_phases": 2,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2, 3],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        #"device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "device": "cpu", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./CheckpointsMP_BOOM/",
        "test_load_weight": "ckpt_100000_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledImgName": "illite_mp.npy",
        "num_samples": 4,
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
