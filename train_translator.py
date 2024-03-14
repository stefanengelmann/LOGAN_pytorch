import os
import torch
from torch import nn
import argparse
from tqdm import tqdm
from itertools import chain

from autoencoder import Autoencoder
from translator_discriminator import Translator, Discriminator
from plyDataloader import plyDataset
from gradient_penalty import GradientPenalty

parser = argparse.ArgumentParser(description='Train Translator')

# Autoencoder model parameters
parser.add_argument('--ae_model_path', type=str, default='weights/best_autoencoder_emd_chair_table.pt', help='path to autoencoder model parameters (default: weights/best_autoencoder_emd_chair_table.pt)')

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 600)')
parser.add_argument('--init_lr', type=float, default=0.002, help='initial learning rate (default: 2e-3)')
parser.add_argument('--alpha_reg', type=float, default=20, help='Scalar weight denoted as alpha in paper (default: 20)')
parser.add_argument('--lambdaGP', type=float, default=10, help='Scalar weight denoted as lambda_2 in paper, used for gradient penalty (default: 10)')
parser.add_argument('--beta_reg', type=float, default=20, help='Scalar weight denoted as beta in paper (default: 20)')
parser.add_argument('--beta_optim', type=float, default=0.5, help='Beta_1 value used in Adam optimizer (default: 0.5)')
parser.add_argument('--D_iter', type=int, default=2, help='Number of times to train the discriminator for every generator iteration (default: 2)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device (default: cuda if available else cpu)')

# Data
parser.add_argument('--root_dir', type=str, default='data/', help='root directory of the dataset (default: data/)')
parser.add_argument('--class_name_A', type=str, default='chair', help='class name A (default: chair)')
parser.add_argument('--class_name_B', type=str, default='table', help='class name B (default: table)')

# Save
parser.add_argument('--save_dir', type=str, default='weights/', help='directory to save model parameters (default: weights/)')
parser.add_argument('--save_models', action='store_true', help='save model parameters after training (default: False)')

args = parser.parse_args()

device = torch.device(args.device)

data_dir = args.root_dir + args.class_name_A + '-'+ args.class_name_B + '/'
print(f"Data directory: {data_dir}")

# Dataloaders
a_train_dataset = plyDataset(data_dir, _class = args.class_name_A, split='train')
a_test_dataset = plyDataset(data_dir, _class = args.class_name_A, split='test')
b_train_dataset = plyDataset(data_dir, _class = args.class_name_B, split='train')
b_test_dataset = plyDataset(data_dir, _class = args.class_name_B, split='test')

print(f"Number of A train samples: {len(a_train_dataset)}")
print(f"Number of B train samples: {len(b_train_dataset)}")
print(f"Number of A test samples: {len(a_test_dataset)}")
print(f"Number of B test samples: {len(b_test_dataset)}")

# Create dataloaders
a_train_dataloader = torch.utils.data.DataLoader(a_train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
b_train_dataloader = torch.utils.data.DataLoader(b_train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)

# TODO: test dataloaders?
print("No validation of the translator for now!")

print(f"Number of A train batches: {len(a_train_dataloader)}")
print(f"Number of B train batches: {len(b_train_dataloader)}")

# Model, Loss, Optimizer
autoencoder = Autoencoder(return_zi=False).to(device).eval()
autoencoder.load_state_dict(torch.load(args.ae_model_path, map_location=device))
encoder = autoencoder.encoder
decoder = autoencoder.decoder

GP = GradientPenalty(lambdaGP=args.lambdaGP)

T_A2B = Translator().to(device)
T_B2A = Translator().to(device)
D_A2B = Discriminator().to(device)
D_B2A = Discriminator().to(device)

optimizer_T = torch.optim.Adam(chain(T_A2B.parameters(),T_B2A.parameters()), lr=args.init_lr, betas=(args.beta_optim, 0.999)) # should it be a list instead of chain?
scheduler_T = torch.optim.lr_scheduler.MultiStepLR(optimizer_T, milestones=[100, 200], gamma=0.5)

optimizer_D_A2B = torch.optim.Adam(D_A2B.parameters(), lr=args.init_lr, betas=(args.beta_optim, 0.999))
optimizer_D_B2A = torch.optim.Adam(D_B2A.parameters(), lr=args.init_lr, betas=(args.beta_optim, 0.999))
scheduler_D_A2B = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_A2B, milestones=[100, 200], gamma=0.5)
scheduler_D_B2A = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_B2A, milestones=[100, 200], gamma=0.5)


# Training
print(f"Training Translator on {args.device}")
for epoch in range(args.epochs):
    T_A2B.train()
    T_B2A.train()
    D_A2B.train()
    D_B2A.train()

    epoch_loss_overall = 0
    epoch_loss_D_A2B = 0
    epoch_loss_D_B2A = 0

    with tqdm(total=min(len(a_train_dataloader),len(b_train_dataloader)), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
        for (data_A, _), (data_B, _) in zip(a_train_dataloader, b_train_dataloader):
            data_A = data_A.to(device)
            data_B = data_B.to(device)

            ########################
            # Train discriminators #
            ########################
            with torch.no_grad():
                data_all = torch.cat([data_A,data_B], dim=0)
                z_all = encoder(data_all)
                
                z_A = z_all[:args.batch_size]
                z_B = z_all[args.batch_size:]

                z_A2B = T_A2B(z_A)
                z_B2A = T_B2A(z_B)

            for _ in range(args.D_iter):
                optimizer_D_A2B.zero_grad()
                optimizer_D_B2A.zero_grad()
                
                # GAN losses:  AB -> B
                logit_A2B = D_A2B(z_A2B)
                logit_B = D_A2B(z_B)

                GP_A2B = GP(D_A2B, z_B, z_A2B)

                loss_D_A2B = -(torch.mean(logit_B) - torch.mean(logit_A2B)) + GP_A2B

                loss_D_A2B.backward()
                optimizer_D_A2B.step()
                epoch_loss_D_A2B += loss_D_A2B.item()

                # GAN losses:  AB -> A
                logit_B2A = D_B2A(z_B2A)
                logit_A = D_B2A(z_A)

                GP_B2A = GP(D_B2A, z_A, z_B2A)

                loss_D_B2A = -(torch.mean(logit_A) - torch.mean(logit_B2A)) + GP_B2A

                loss_D_B2A.backward()
                optimizer_D_B2A.step()
                epoch_loss_D_B2A += loss_D_B2A.item()
            
            ####################
            # Train translator #
            ####################
            optimizer_T.zero_grad()
            
            z_A = z_A.detach() # TODO: is this necessary?
            z_B = z_B.detach() # TODO: is this necessary?

            z_A2B = T_A2B(z_A)
            z_B2A = T_B2A(z_B)
            z_B2B = T_A2B(z_B)
            z_A2A = T_B2A(z_A)
            z_A2B2A = T_B2A(z_A2B)
            z_B2A2B = T_A2B(z_B2A)
            
            # WGAN losses
            with torch.no_grad():
                logit_A2B = D_A2B(z_A2B)
                logit_B2A = D_B2A(z_B2A)

            loss_WGAN_A2B = -torch.mean(logit_A2B)
            loss_WGAN_B2A = -torch.mean(logit_B2A)

            # Feature preservation losses
            loss_FP_A2B = nn.functional.l1_loss(z_B2B, z_B) # TODO: check if it also averages over the batch
            loss_FP_B2A = nn.functional.l1_loss(z_A2A, z_A) # TODO: check if it also averages over the batch

            # Cycle consistency loss  
            loss_cycle = nn.functional.l1_loss(z_A2B2A, z_A) + nn.functional.l1_loss(z_B2A2B, z_B) # TODO: check if it also averages over the batch

            # Translation losses
            loss_A2B = loss_WGAN_A2B + args.alpha_reg * loss_FP_A2B
            loss_B2A = loss_WGAN_B2A + args.alpha_reg * loss_FP_B2A

            loss_overall = loss_A2B + loss_B2A + args.beta_reg * loss_cycle

            loss_overall.backward()
            optimizer_T.step()
            epoch_loss_overall += loss_overall.item()

            pbar.update(1)
        
        epoch_loss_overall /= min(len(a_train_dataloader), len(b_train_dataloader))
        epoch_loss_D_A2B /= min(len(a_train_dataloader), len(b_train_dataloader))*args.D_iter
        epoch_loss_D_B2A /= min(len(a_train_dataloader), len(b_train_dataloader))*args.D_iter
        pbar.set_postfix({"Overall": f"{epoch_loss_overall:.2f}", "D_A2B": f"{epoch_loss_D_A2B:.2f}", "D_B2A": f"{epoch_loss_D_B2A:.2f}"})

        scheduler_D_A2B.step()
        scheduler_D_B2A.step()
        scheduler_T.step()

print("Training complete")

if args.save_models:
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(T_A2B.state_dict(), args.save_dir + f"best_translator_{args.class_name_A}_to_{args.class_name_B}.pt")
    torch.save(T_B2A.state_dict(), args.save_dir + f"best_translator_{args.class_name_B}_to_{args.class_name_A}.pt")
    torch.save(D_A2B.state_dict(), args.save_dir + f"best_discriminator_{args.class_name_A}_to_{args.class_name_B}.pt")
    torch.save(D_B2A.state_dict(), args.save_dir + f"best_discriminator_{args.class_name_B}_to_{args.class_name_A}.pt")
