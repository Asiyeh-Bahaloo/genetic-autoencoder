import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedLinearLayer(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(out, inp))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.param)

    def forward(self, input, transpose):
        if transpose is False:
            output = F.linear(input, self.param)
        else:
            output = F.linear(input, self.param.t())
        return output


class SingleTiedAutoEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layers = TiedLinearLayer(inp, out)

    def forward(self, input):
        x = torch.flatten(input, start_dim=1)
        x = self.layers(x, transpose=False)
        x = self.layers(x, transpose=True)
        return x


class TiedAutoEncoder(nn.Module):
    def __init__(self, shape_list, nonlinearity=torch.relu):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nonlinearity = nonlinearity
        self.shape_list = shape_list
        for i in range(len(self.shape_list) - 1):
            self.layers.append(
                TiedLinearLayer(self.shape_list[i], self.shape_list[i + 1])
            )

    def forward(self, input):
        x = torch.flatten(input, start_dim=1)
        # encode
        for layer in self.layers:
            x = layer(x, transpose=False)
            x = self.nonlinearity(x)
        encoded_feats = x.detach().clone()
        # decode
        for i, layer in sorted(enumerate(self.layers), reverse=True):
            x = layer(x, transpose=True)
            if i != 0:  # if it's not the last layer
                x = self.nonlinearity(x)
        reconstructed_output = x
        return encoded_feats, reconstructed_output


class GeneticTiedAutoEncoder(TiedAutoEncoder):
    def __init__(self, shape_list, nonlinearity=torch.relu):
        super().__init__(shape_list, nonlinearity)

    @torch.no_grad()
    def crossover(self, parent, layer_key, p):
        parent1 = self.state_dict()[layer_key]
        parent2 = parent.state_dict()[layer_key]
        assert parent1.shape == parent2.shape, "shape of parents should match"

        weights_mask = torch.FloatTensor(parent1.shape).uniform_(0, 1) <= p
        new_chromosome = parent1 * weights_mask + parent2 * weights_mask.logical_not()

        child = GeneticTiedAutoEncoder(self.shape_list, self.nonlinearity)
        child.state_dict()[layer_key] = weights_mask
        return child

    @torch.no_grad()
    def mutation(self, layer_key, p):
        chromosome = self.state_dict()[layer_key]
        weights_mask = torch.FloatTensor(chromosome.shape).uniform_(0, 1) > p
        self.state_dict()[layer_key] = chromosome * weights_mask
