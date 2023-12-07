import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50, resnet34



class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(nn.Module):
    def __init__(self, cfg):

        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super(SimCLR, self).__init__()

        self.arch = cfg['arch']

        self.first_conv = cfg['first_conv']
        self.maxpool1 = cfg['maxpool1']

        self.hidden_mlp = cfg['hidden_mlp']
        self.feat_dim = cfg['feat_dim']

        self.projection = Projection(input_dim=512, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

        self.encoder = self.init_model()


    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet34":
            backbone = resnet34
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, img1, img2):

        # bolts resnet returns a list
        h1 = self.encoder(img1)[-1]
        h2 = self.encoder(img2)[-1]

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return z1, z2

    def loss_function(self, out_1, out_2, temperature, eps=1e-6):
        """
        nt_xent_loss function

        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

if __name__ == "__main__":

    import yaml
    import argparse

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/simCLR.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    ssl = SimCLR(config['model'])

    img1 = torch.randn((128, 3, 256, 256))
    img2 = torch.randn((128, 3, 256, 256))

    #current_device = torch.device("cuda")
    #x = x.to(current_device)

    z1, z2 = ssl(img1, img2)

    print('xhat shape = {}'.format(img1.shape))

    print('latent_z1 = {}'.format(z1.shape))
    print('latent_z2 = {}'.format(z2.shape))

    #print('recons_features shape = {}'.format(recons_features.shape))
    #print('input_features shape = {}'.format(input_features.shape))

    print('Test done')
