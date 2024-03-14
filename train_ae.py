import os
import torch
from torch.utils.data import ConcatDataset
import argparse
from tqdm import tqdm

from autoencoder import Autoencoder
from losses import ChamferLoss, EMDLoss
from plyDataloader import plyDataset

parser = argparse.ArgumentParser(description='Train Autoencoder')

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 400)')
#parser.add_argument('--latent_size', type=int, default=256, help='latent size (default: 256)') # hardcoded in autoencoder for now
#parser.add_argument('--input_size', type=int, default=2048, help='input size (default: 2048)') # hardcoded in autoencoder for now
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate (default: 5e-4)')
parser.add_argument('--ae_loss', type=str, default='emd', choices=['chamfer', 'emd'], help='loss function (default: emd)')
parser.add_argument('--lambda_zi', type=float, default=0.1, help='Scalar weight denoted as lambda_1 in paper (default: 0.1)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device (default: cuda if available else cpu)')

# Data
parser.add_argument('--root_dir', type=str, default='data/', help='root directory of the dataset (default: data/)')
parser.add_argument('--class_name_A', type=str, default='chair', help='class name A (default: chair)')
parser.add_argument('--class_name_B', type=str, default='table', help='class name B (default: table)')

# Load
parser.add_argument('--load_pretrained', type=str, default=None, help='Path to pretrained model to be loaded (default: None)')

# Save
parser.add_argument('--save_dir', type=str, default='weights/', help='directory to save model parameters (default: weights/)')
parser.add_argument('--save_best', action='store_true', help='save the best model parameters (default: False)')

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

# Concatenate the datasets
train_dataset = ConcatDataset([a_train_dataset, b_train_dataset])
test_dataset = ConcatDataset([a_test_dataset, b_test_dataset])

print(f"Number of train samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Create the collective dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=False)

print(f"Number of train batches: {len(train_dataloader)}")
print(f"Number of test batches: {len(test_dataloader)}")

# Model, Loss, Optimizer
autoencoder = Autoencoder().to(device)
if args.load_pretrained:
    autoencoder.load_state_dict(torch.load(args.load_pretrained))
    print(f"Pretrained model loaded from {args.load_pretrained}")
ae_loss = {'chamfer': ChamferLoss(), 'emd': EMDLoss()}[args.ae_loss]
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.learning_rate)

# Training
print(f"Training Autoencoder on {args.device} with {args.ae_loss} loss")
best_test_loss = 1e9
for epoch in range(args.epochs):
    autoencoder.train()
    train_loss = 0
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
        for data, _ in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            rec_z, rec_zis = autoencoder(data)
            loss = ae_loss(rec_z, data) + args.lambda_zi * sum([ae_loss(rec_zi, data) for rec_zi in rec_zis])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.update(1)
        train_loss /= len(train_dataloader)
        pbar.set_postfix({"Train Loss": f"{train_loss:.4f}"})

    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_dataloader:
            data = data.to(device)
            rec_z, rec_zis = autoencoder(data)
            loss = ae_loss(data, rec_z) + args.lambda_zi * sum([ae_loss(data, rec_zi) for rec_zi in rec_zis])
            test_loss += loss.item()
        test_loss /= len(test_dataloader)
        if test_loss < best_test_loss and args.save_best:
            best_test_loss = test_loss
            best_state_dict = autoencoder.state_dict()
            best_epoch = epoch
        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {test_loss:.4f}")

if args.save_best:
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(best_state_dict, args.save_dir + f"best_autoencoder_{args.ae_loss}_{args.class_name_A}_{args.class_name_B}.pt")

print("Training complete")