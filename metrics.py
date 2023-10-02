import os
import numpy as np
import torch
import lpips
# from chamferdist import ChamferDistance
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class Meter:
    def __init__(self, name):
        self.V = 0
        self.N = 0
        self.name = name

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, self.name), self.measure(), global_step)

    def report(self):
        return f"{self.name} = {self.measure():.6f}"

class myPSNRMeter(Meter):
    def __init__(self, name, data_range=None, device='cpu'):
        super().__init__(name=name)

    def update(self, preds, truths, *args):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        psnr = 10 * np.log10(((2*args[0]) ** 2 if len(args)>0 else 1) / np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

class PSNRMeter(Meter):
    def __init__(self, name, data_range=None, device='cpu'):
        super().__init__(name=name)
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def update(self, preds, truths, *args):
        psnr_value = self.psnr(preds, truths)

        self.V += psnr_value
        self.N += 1

class SSIMMeter(Meter):
    def __init__(self, name, data_range=None):
        super().__init__(name=name)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def update(self, preds, truths, *args):
        ssim_value = self.ssim(preds.unsqueeze(2).permute(2,3,0,1), truths.unsqueeze(2).permute(2,3,0,1))

        self.V += ssim_value
        self.N += 1

class LPIPSMeter(Meter):
    def __init__(self, name, net="alex", device=None):
        super().__init__(name=name)
        self.net = net
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.unsqueeze(2).permute(2,3,0,1).contiguous() / inp.abs().max()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths, *args):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(
            truths, preds
        ).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

class VarDiffMeter(Meter):
    def __init__(self, name):
        super().__init__(name=name)

    def update(self, preds, truths, *args):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        vardiff = np.abs(preds.var() - truths.var())

        self.V += vardiff
        self.N += 1

class MAEMeter(Meter):
    def __init__(self, name):
        super().__init__(name=name)

    def update(self, preds, truths, *args):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        mae = np.mean(np.abs(preds - truths))

        self.V += mae
        self.N += 1

class ChamferDistMeter(Meter):
    def __init__(self, name, num_timesteps, testskip=1, device=None):
        super().__init__(name=name)
        self.device = (device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.ptcloud_preds = torch.empty((0, 3)).to(self.device)
        self.ptcloud_target = torch.empty((0, 3)).to(self.device)
        self.chamferdist = ChamferDistance()
        # self.ctr = 0
        self.timesteps = torch.linspace(0.0, 1.0, num_timesteps)
        self.testskip = testskip

    def init_ptclouds(self):
        self.ptcloud_preds = torch.empty((0,3)).to(self.device)
        self.ptcloud_target = torch.empty((0,3)).to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0
        self.init_ptclouds()
        # self.ctr = 0

    def update(self, preds, truths, *args):
        # build preds
        pred_nz_ids = preds.squeeze().nonzero()
        pred_ev_ptcloud = torch.hstack((pred_nz_ids, args[0]*torch.ones((pred_nz_ids.shape[0], 1)).to(preds.device)))
        self.ptcloud_preds = torch.vstack((self.ptcloud_preds, pred_ev_ptcloud))

        # build target
        target_nz_ids = truths.reshape(preds.shape).squeeze().nonzero()
        target_ptcloud = torch.hstack((target_nz_ids, args[0]*torch.ones((target_nz_ids.shape[0], 1)).to(preds.device)))
        self.ptcloud_target = torch.vstack((self.ptcloud_target, target_ptcloud))

        # if reached end of a viewpoint's evaluations, then calculate the metric and update
        if ((args[1]+self.testskip) // self.timesteps.shape[0]) > (args[1] // self.timesteps.shape[0]):
        # if args[0][1]==None or args[0][1] <= args[0][0]:

            cdist = self.chamferdist(self.ptcloud_preds.unsqueeze(0), self.ptcloud_target.unsqueeze(0))

            self.V += cdist
            self.N += 1
        
            # reset pointclouds
            self.init_ptclouds()

            # reset ctr
            # self.ctr = -self.testskip

        # self.ctr += self.testskip
