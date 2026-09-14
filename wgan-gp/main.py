import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader, get_image_folder_dataloader
from models import Generator, Discriminator
from training import Trainer

data_loader = get_image_folder_dataloader("/export/home/aaouf/workspace/2d_images/banderabrown_2D_dataset/", batch_size=64)
img_size = (256, 256, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=64)
discriminator = Discriminator(img_size=img_size, dim=64)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.5, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 1500
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'banderabrown__model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
