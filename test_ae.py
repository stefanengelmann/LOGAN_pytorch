import torch
from torch.utils.data import ConcatDataset
import argparse
from tqdm import tqdm
import k3d

from autoencoder import Autoencoder
from losses import ChamferLoss, EMDLoss
from plyDataloader import plyDataset

parser = argparse.ArgumentParser(description='Evaluate Autoencoder')

# Autoencoder model parameters
parser.add_argument('--ae_model_path', type=str, default='weights/best_autoencoder_emd_chair_table.pt', help='path to autoencoder model parameters (default: weights/best_autoencoder_emd_chair_table.pt)')

# Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--ae_loss', type=str, default='emd', choices=['chamfer', 'emd'], help='loss function (default: emd)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device (default: cuda if available else cpu)')
parser.add_argument('--visualize', action='store_true', help='visualize the results (default: False)')

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

# Concatenate the datasets
test_dataset = ConcatDataset([a_test_dataset, b_test_dataset])

print(f"Number of test samples: {len(test_dataset)}")

# Create the collective dataloader
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=False)

print(f"Number of test batches: {len(test_dataloader)}")

# Model, Loss, Optimizer
autoencoder = Autoencoder(return_zi=False).to(device)
autoencoder.load_state_dict(torch.load(args.ae_model_path, map_location=device))
ae_loss = {'chamfer': ChamferLoss(), 'emd': EMDLoss()}[args.ae_loss]

# Evaluation
autoencoder.eval()
test_loss = 0
with torch.no_grad():
    with tqdm(total=len(test_dataloader), desc=f"Evaluating Autoencoder on {args.device} with {args.ae_loss} loss") as pbar:
        for i, (data, filename) in enumerate(test_dataloader):
            data = data.to(device)
            rec_z = autoencoder(data)
            loss = ae_loss(data, rec_z)/2048 if args.ae_loss == 'emd' else ae_loss(data, rec_z)/2048/len(data)
            test_loss += loss.item()

            if args.visualize:
                data = data.cpu().numpy()
                rec_z = rec_z.cpu().numpy()
                for j in range(args.batch_size):
                    plot = k3d.plot()
                    plt_points = k3d.points(data[j], point_size=0.04, color=0x00ff00, name='Original')
                    rec_z[j][:,1] += 1
                    rec_points = k3d.points(rec_z[j], point_size=0.04, color=0xff0000, name='Reconstructed')
                    plot += plt_points
                    plot += rec_points
                    with open(filename[j][:-3]+'html','w', encoding="utf-8") as fp:
                        fp.write(plot.get_snapshot())
                    input("Press Enter to continue...")

            pbar.update(1)
        test_loss /= len(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}")





