import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule

class Autoencoder(nn.Module):
    def __init__(self, return_zi=True):
        super().__init__()
        self.return_zi = return_zi
        self.encoder = Encoder(return_zi=return_zi)
        self.decoder = Decoder()

    def forward(self, x):
        if self.return_zi:
            z, z_is = self.encoder(x)
            rec_z = self.decoder(z)
            rec_zis = []
            for z_i in z_is:
                rec_z_i = self.decoder(z_i)
                rec_zis.append(rec_z_i)
            return rec_z, rec_zis
        else:
            z = self.encoder(x)
            rec_z = self.decoder(z)
            return rec_z

class Encoder(nn.Module):
    def __init__(self, return_zi=True):
        super().__init__()
        self.return_zi = return_zi

        self.SA1 = PointnetSAModule(npoint=512,radius=0.1,nsample=64, mlp=[0,64,64,128])
        self.SA2 = PointnetSAModule(npoint=256,radius=0.2,nsample=64, mlp=[128,128,128,256])
        self.SA3 = PointnetSAModule(npoint=128,radius=0.3,nsample=64, mlp=[256,256,256,256])
        self.SA4 = PointnetSAModule(npoint=32,radius=0.4,nsample=64, mlp=[256,256,256,256])

        self.MLP1 = Encoder_with_convs_and_symmetry(128,n_filters=[128,128,64])
        self.MLP2 = Encoder_with_convs_and_symmetry(256,n_filters=[256,256,64])
        self.MLP3 = Encoder_with_convs_and_symmetry(256,n_filters=[256,256,64])
        self.MLP4 = Encoder_with_convs_and_symmetry(256,n_filters=[256,256,64])


    def forward(self, x):
        l0_xyz = x
        l0_points = None

        l1_xyz, l1_points = self.SA1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.SA2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.SA4(l3_xyz, l3_points)

        output_1 = self.MLP1(l1_points)
        output_2 = self.MLP2(l2_points)
        output_3 = self.MLP3(l3_points)
        output_4 = self.MLP4(l4_points)

        z = torch.cat([output_1, output_2, output_3, output_4], dim=1)

        if self.return_zi:
            z_is = []
            for i in range(4):
                z_i = torch.zeros_like(z)
                z_i[:, 64*i:64*(i+1)] = [output_1, output_2, output_3, output_4][i]
                z_is.append(z_i.unsqueeze(-1))

            return z.unsqueeze(-1), z_is
        else:
            return z.unsqueeze(-1)
    
class Encoder_with_convs_and_symmetry(nn.Module):
    def __init__(self, input_size, n_filters: list):
        super(Encoder_with_convs_and_symmetry, self).__init__()

        if len(n_filters) != 3:
            raise ValueError("n_filters must be a list of length 3")
        
        self.layer = nn.Sequential(
            nn.Conv1d(input_size, n_filters[0], 1),
            nn.BatchNorm1d(n_filters[0]),
            nn.ReLU(),
            nn.Conv1d(n_filters[0], n_filters[1], 1),
            nn.BatchNorm1d(n_filters[1]),
            nn.ReLU(),
            nn.Conv1d(n_filters[1], n_filters[2], 1),
            nn.BatchNorm1d(n_filters[2]),
            nn.ReLU(),
        )
        
    def forward(self, feature):
        feature = self.layer(feature)
        feature = feature.max(2)[0]
        return feature
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 6144, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 2048, 3)
        return x


