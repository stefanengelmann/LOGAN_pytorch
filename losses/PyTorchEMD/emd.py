"""Modified from https://github.com/daerduoCarey/PyTorchEMD"""
import torch
import emd_cuda
from torch.nn import Module

class EMDLoss(Module):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, n, 3)
        xyz2 (torch.Tensor): (b, n, 3)

    Returns:
        cost (torch.Tensor): (1,)

    """
    def __init__(self):
        super().__init__()
        self.emd_function = EarthMoverDistanceFunction.apply

    def forward(self, xyz1, xyz2):
        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)

        return self.emd_function(xyz1, xyz2).mean()


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2

