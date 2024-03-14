import torch
import numpy as np
import k3d
import argparse

from autoencoder import Autoencoder
from translator_discriminator import Translator
from plyDataloader import plyDataset

parser = argparse.ArgumentParser(description='LOGAN Inference')

# Model parameters
parser.add_argument('--ae_model_path', type=str, default='weights/best_autoencoder_emd_chair_table.pt', help='path to autoencoder model parameters (default: weights/best_autoencoder_emd_chair_table.pt)')
parser.add_argument('--t_a2b_model_path', type=str, default='weights/best_translator_chair_to_table.pt', help='path to the T_A2B model parameters (default: weights/best_translator_chair_to_table.pt)')
parser.add_argument('--t_b2a_model_path', type=str, default='weights/best_translator_table_to_chair.pt', help='path to the T_B2A model parameters (default: weights/best_translator_table_to_chair.pt)')

# Hyperparameters
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device (default: cuda if available else cpu)')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for inference (default: 32)')

# Data
parser.add_argument('--root_dir', type=str, default='data/', help='root directory of the dataset (default: data/)')
parser.add_argument('--class_name_A', type=str, default='chair', help='class name A (default: chair)')
parser.add_argument('--class_name_B', type=str, default='table', help='class name B (default: table)')

args = parser.parse_args()

device = torch.device(args.device)

data_dir = args.root_dir + args.class_name_A + '-'+ args.class_name_B + '/'
print(f"Data directory: {data_dir}")

# Dataloaders
a_test_dataset = plyDataset(data_dir, _class = args.class_name_A, split='test')
b_test_dataset = plyDataset(data_dir, _class = args.class_name_B, split='test')

print(f"Number of A test samples: {len(a_test_dataset)}")
print(f"Number of B test samples: {len(b_test_dataset)}")

# Create dataloaders
a_test_dataloader = torch.utils.data.DataLoader(a_test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=False)
b_test_dataloader = torch.utils.data.DataLoader(b_test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=False)

# Initialize models
autoencoder = Autoencoder(return_zi=False).to(device).eval()
autoencoder.load_state_dict(torch.load(args.ae_model_path, map_location=device))
encoder = autoencoder.encoder
decoder = autoencoder.decoder

T_A2B = Translator().to(device).eval()
T_A2B.load_state_dict(torch.load(args.t_a2b_model_path, map_location=device))
T_B2A = Translator().to(device).eval()
T_B2A.load_state_dict(torch.load(args.t_b2a_model_path, map_location=device))


# Inference
with torch.no_grad():
    for i, (data_A, filename) in enumerate(a_test_dataloader):
        data_A = data_A.to(device)
        z_A = encoder(data_A)
        z_A2B = T_A2B(z_A)
        data_A2B = decoder(z_A2B)
        data_A2B = data_A2B.cpu().numpy()
        data_A = data_A.cpu().numpy()
        for j in range(args.batch_size):
            plot = k3d.plot()
            plt_points = k3d.points(data_A[j], point_size=0.04, color=0x00ff00, name=args.class_name_A)
            data_A2B[j][:,1] += 1
            rec_points = k3d.points(data_A2B[j], point_size=0.04, color=0xff0000, name='translated '+args.class_name_B)
            plot += plt_points
            plot += rec_points
            with open(filename[j][:-3]+'html','w', encoding="utf-8") as fp:
                fp.write(plot.get_snapshot())
            input("Press Enter to continue...")

    for i, (data_B, filename) in enumerate(b_test_dataloader):
        data_B = data_B.to(device)
        z_B = encoder(data_B)
        z_B2A = T_B2A(z_B)
        data_B2A = decoder(z_B2A)
        data_B2A = data_B2A.cpu().numpy()
        data_B = data_B.cpu().numpy()
        for j in range(args.batch_size):
            plot = k3d.plot()
            plt_points = k3d.points(data_B[j], point_size=0.04, color=0x00ff00, name=args.class_name_B)
            data_B2A[j][:,1] += 1
            rec_points = k3d.points(data_B2A[j], point_size=0.04, color=0xff0000, name='translated '+args.class_name_A)
            plot += plt_points
            plot += rec_points
            with open(filename[j][:-3]+'html','w', encoding="utf-8") as fp:
                fp.write(plot.get_snapshot())
            input("Press Enter to continue...")

