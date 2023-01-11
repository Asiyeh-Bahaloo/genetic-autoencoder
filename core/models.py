import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedLinearLayer(nn.Module):
    """A tied linear layer with Xavier initialization."""

    def __init__(self, inp, out):

        super().__init__()
        self.param = nn.Parameter(torch.zeros(out, inp))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters to have Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.param)

    def forward(self, input, transpose):
        """The forward function for the layer. The transpose parameter indicates whether to use
         the layer in the forward path of the decoder or encoder.

        Parameters
        ----------
        input : torch.tensor
            The input tensor for the layer.
        transpose : bool
            If False, the layer is in the decoder part; otherwise, the layer is in the encoder
            part, and the weights need to be transposed.

        Returns
        -------
        torch.tensor
            The output of the layer.
        """
        if transpose is False:
            output = F.linear(input, self.param)
        else:
            output = F.linear(input, self.param.t())
        return output


class SingleTiedAutoEncoder(nn.Module):
    """A one layer tied auto encoder using TiedLinearLayer."""

    def __init__(self, inp, out):
        super().__init__()
        self.layers = TiedLinearLayer(inp, out)

    def forward(self, input):
        """The forward function with one single layer for decoder and encoder.

        Parameters
        ----------
        input : torch.tensor
            The input tensor for the module.

        Returns
        -------
        torch.tensor
            The output of the module.
        """
        x = torch.flatten(input, start_dim=1)
        x = self.layers(x, transpose=False)
        x = self.layers(x, transpose=True)
        return x


class TiedAutoEncoder(nn.Module):
    """A tied auto encoder with arbitrary layer numbers, layer shapes, and nonlinearity between
    them. the building block is TiedLinearLayer.
    """

    def __init__(self, shape_list, nonlinearity=torch.relu):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nonlinearity = nonlinearity
        self.shape_list = shape_list

        # add TiedLinearLayer with desired count and shapes to the model
        for i in range(len(self.shape_list) - 1):
            self.layers.append(
                TiedLinearLayer(self.shape_list[i], self.shape_list[i + 1])
            )

    def forward(self, input):
        """The forward function that creates the autoencoder output.

        Parameters
        ----------
        input : torch.tensor
            The input tensor for the autoencoder.

        Returns
        -------
        torch.tensor, torch.tensor
            The encoded features, the output of the autoencoder.
        """
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
    """A TiedAutoEncoder which enables two primary genetic functions."""

    def __init__(self, shape_list, nonlinearity=torch.relu):
        super().__init__(shape_list, nonlinearity)

    @torch.no_grad()
    def crossover(self, parent, layer_key, p):
        """Create a new offspring by applying one-point crossover to parents with probability p.
        Parameters
        ----------
        parent : TiedAutoEncoder
            The second applicant for the crossover.
        layer_key : str
            The name of the layer that contains the learning chromosome.
        p : float
            The probability of crossover. Means how many of the weights should taken from one parent.

        Returns
        -------
        GeneticTiedAutoEncoder
            A new model that contains the offspring's weights for the particular chromosome.
        """
        parent1 = self.state_dict()[layer_key]
        parent2 = parent.state_dict()[layer_key]
        assert parent1.shape == parent2.shape, "shape of parents should match"

        weights_mask = torch.FloatTensor(parent1.shape).uniform_(0, 1) <= p
        new_chromosome = parent1 * weights_mask + parent2 * weights_mask.logical_not()

        child = GeneticTiedAutoEncoder(self.shape_list, self.nonlinearity)
        child.state_dict()[layer_key] = new_chromosome
        return child

    @torch.no_grad()
    def mutation(self, layer_key, p):
        """Change (in place) the weights of the chromosomes to zero randomly with probability p.

        Parameters
        ----------
        layer_key : str
            The name of the layer that contains the learning chromosome.
        p : float
            The probability for each gene to turn to zero.
        """
        chromosome = self.state_dict()[layer_key]
        weights_mask = torch.FloatTensor(chromosome.shape).uniform_(0, 1) > p
        self.state_dict()[layer_key] = chromosome * weights_mask
