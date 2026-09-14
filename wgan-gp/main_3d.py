# main_3d.py
import torch
import torch.optim as optim

from dataloaders_3d import get_npy_volume_dataloader
from models_3d import Generator3D, Discriminator3D
from training_3d import Trainer3D


def main():
    data_root = "/export/home/aaouf/workspace/images/datasets_3d/bentheimer_bin"
    batch_size = 6  # 3D is heavy; start small
    data_loader = get_npy_volume_dataloader(
        data_root,
        batch_size=batch_size,
    )

    vol_size = (64, 64, 64)
    generator = Generator3D(vol_size=vol_size, latent_dim=100, dim=64, out_channels=1)
    discriminator = Discriminator3D(vol_size=vol_size, dim=64, in_channels=1)

    print(generator)
    print(discriminator)

    lr = 1e-4
    betas = (0.5, 0.99)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    epochs = 1500
    trainer = Trainer3D(
        generator,
        discriminator,
        G_optimizer,
        D_optimizer,
        use_cuda=torch.cuda.is_available(),
        checkpoint_dir="./checkpoints_3d",
        save_png_every=10,
        checkpoint_every=10,
        png_grid_size=32,
        png_nrow=8,
        slice_axis="z",  # "z" (axial), "y", or "x"
    )

    trainer.train(data_loader, epochs, save_training_gif=True)

    name = "banderabrown__model_3d"
    torch.save(trainer.G.state_dict(), "./gen_" + name + ".pt")
    torch.save(trainer.D.state_dict(), "./dis_" + name + ".pt")


if __name__ == "__main__":
    main()
