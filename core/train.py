import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms


import numpy as np
import hydra
from omegaconf import DictConfig


# import matplotlib.pyplot as plt
# from genetic_autoencoder.core.models import *
from models import TiedAutoEncoder, GeneticTiedAutoEncoder


def train(model, device, train_loader, optimizer, epoch, loss_fn=F.mse_loss):
    """Train the model with stochastic gradient descent for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model we want to train.
    device : torch.device
        Specify which device (cpu or GPU) should be used for training.
    train_loader : torch.DataLoader
        Data loader for the train dataset.
    optimizer : torch.optimizer
        Optimizer for the model.
    epoch : int
        Epoch number.
    loss_fn : function, optional
        Loss function, by default F.mse_loss
    """
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        _, reconstructed_output = model(data)
        target = torch.flatten(data, start_dim=1)
        loss = loss_fn(reconstructed_output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(
                f"""Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}
                ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"""
            )


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """Evaluate the model and calculate the MSE loss.

    Parameters
    ----------
    model : nn.Module_
        Model for the evaluation purposes.
    data_loader : torch.DataLoader
        Data loader for a specific dataset (train or test).
    device : torch.device
        Specify which device (cpu or GPU) should be used for training.

    Returns
    -------
    float
        MSE loss of the model on the whole dataset.
    """
    val_loss = 0
    for data, _ in data_loader:
        data = data.to(device)
        _, reconstructed_output = model(data)
        target = torch.flatten(data, start_dim=1)
        loss = F.mse_loss(reconstructed_output, target)
        val_loss += loss

    validation_loss = val_loss / len(data_loader)
    return validation_loss


def layer_wise_train(model, device, train_loader, lr, epoch, loss_fn=F.mse_loss):
    """Train the whole (TiedAutoEncoder) model in a layer-wise manner using the train function
    at each epoch.

    Parameters
    ----------
    model : TiedAutoEncoder
        The model we want to train.
    device : torch.device
        Specify which device (cpu or GPU) should be used for training.
    train_loader : torch.DataLoader
        Data loader for the train dataset.
    lr : float
        Learning rate for optimizer.
    epoch : int
        The total number of train epochs.
    loss_fn : function, optional
        Loss function, by default F.mse_loss
    """
    shape_list = model.shape_list  # shape_list = [784, 500, 250, 100, 50]
    current_shape_list = []
    weight_state_dict = {}

    for layer_shape in shape_list:
        current_shape_list.append(layer_shape)
        if len(current_shape_list) < 2:
            continue
        # recreate model to have just one learnable layer
        model = TiedAutoEncoder(current_shape_list, nonlinearity=torch.relu)
        print("training .... \n", model)
        # load prev weights
        model.load_state_dict(weight_state_dict, strict=False)

        # freeze network except last layer
        for param in list(model.parameters())[:-1]:
            param.requires_grad = False
        # train the model
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        for epoch in range(1, epoch + 1):
            train(model, device, train_loader, optimizer, epoch, loss_fn)
        # update weights
        weight_state_dict = model.state_dict()


def train_genetic_model(
    shape_list,
    train_loader,
    pop_size,
    generations,
    device,
    prev_weights=None,
    finetune_epoch=3,
    finetune_lr=0.01,
    loss_fn=F.mse_loss,
):
    """Train the last layer in a GeneticTiedAutoEncoder model using a genetic algorithm.

    Parameters
    ----------
    shape_list : python list
        A list containing the shape of distinct autoencoder layers.
    train_loader : torch.DataLoader
        Data loader for the train dataset.
    pop_size : int
        Population size of chromosomes.
    generations : int
        The total training generations.
    device : torch.device
        Specify which device (cpu or GPU) should be used for training.
    prev_weights : dict, optional
        Previous layer weights that should be loaded into the model, by default None
    finetune_epoch : int, optional
        The number of epochs we finetune chromosomes using the stochastic gradient descent
        algorithm, by default 3
    finetune_lr : float, optional
        Learning rate for finetuning, by default 0.01
    loss_fn : function, optional
        Loss function for finetuning, by default F.mse_loss


    Returns
    -------
    GeneticTiedAutoEncoder
        A GeneticTiedAutoEncoder containing the best learned weights.
    """

    # Generate choromosomes
    print(f"\nGenerated {pop_size} models with {shape_list} layer size! ")
    models = [
        GeneticTiedAutoEncoder(shape_list, nonlinearity=torch.relu)
        for i in range(pop_size)
    ]

    # Load prevoius layer weights
    print()
    if prev_weights is not None:
        for model in models:
            model.load_state_dict(prev_weights, strict=False)
        shapes = []
        for key_weight in prev_weights:
            shapes.append(prev_weights[key_weight].shape)
        print(f"\nLoaded prevoius layer weights with shape {shapes}")

    layer_key = next(reversed(models[0].state_dict()))  # last layer key
    print(
        f"\nTraining {layer_key} with shape of {models[0].state_dict()[layer_key].shape} : "
    )

    for g in range(generations):
        print(f"\nGeneration {g + 1}: \n")
        print("Calculating fitness for each chromosome...")
        # fitness
        fitness = []
        for model in models:
            fitness.append(1 / evaluate_loss(model, train_loader, device))

        # Select 5 best models (/chromosomes)
        fit_arg = np.argsort(fitness)
        selected_models = [models[f] for f in fit_arg[: pop_size // 2]]
        print(f"Selected {pop_size // 2} Top best chromosomes")

        # finetune selected models
        print("Finetuning selected models...\n")
        for model in selected_models:
            # freeze network except the chromosome layer
            for name, param in model.named_parameters():
                if name == layer_key:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # train the model
            optimizer = optim.Adadelta(model.parameters(), lr=finetune_lr)
            for epoch in range(1, finetune_epoch + 1):
                train(model, device, train_loader, optimizer, epoch, loss_fn)

        # cross over and mutation
        print(
            f"Generate {pop_size // 2} other chromosomes with Cross-Over and Mutation\n"
        )
        for _ in range(pop_size // 2):
            mom_idx = np.random.randint(low=0, high=pop_size // 2)
            dad_idx = np.random.randint(low=0, high=pop_size // 2)
            child = selected_models[mom_idx].crossover(
                selected_models[dad_idx], layer_key, p=0.8
            )
            child.mutation(layer_key, p=0.01)
            selected_models.append(child)
        models = selected_models

    fitness = []
    print("Selecting best chromosome as answer...")
    # fitness
    for model in models:
        fitness.append(1 / evaluate_loss(model, train_loader, device))
    fit_arg = np.argsort(fitness)
    best_model = models[fit_arg[-1]]
    print("Finished")
    return best_model


def layerwise_genetic_train(
    shape_list,
    train_loader,
    pop_size,
    generations,
    device,
    finetune_epoch=3,
    finetune_lr=0.01,
    loss_fn=F.mse_loss,
):
    """Builds and trains the whole GeneticTiedAutoEncoder model in a layer-wise manner using
    the genetic algorithm.

    Parameters
    ----------
    shape_list : python list
        A list containing the shape of distinct autoencoder layers.
    train_loader : torch.DataLoader
        Data loader for the train dataset.
    pop_size : int
        Population size of chromosomes.
    generations : int
        The total training generations.
    device : torch.device
        Specify which device (cpu or GPU) should be used for training.
    finetune_epoch : int, optional
        The number of epochs we finetune chromosomes using the stochastic gradient descent
        algorithm, by default 3
    finetune_lr : float, optional
        Learning rate for finetuning, by default 0.01
    loss_fn : function, optional
        Loss function for finetuning, by default F.mse_loss


    Returns
    -------
    GeneticTiedAutoEncoder
       A GeneticTiedAutoEncoder containing the best learned weights.
    """
    current_shape_list = []
    weight_state_dict = None

    for layer_shape in shape_list:
        current_shape_list.append(layer_shape)
        if len(current_shape_list) < 2:
            continue

        model = train_genetic_model(
            current_shape_list,
            train_loader,
            pop_size,
            generations,
            device,
            weight_state_dict,
            finetune_epoch,
            finetune_lr,
            loss_fn,
        )
        # update the previous weights
        weight_state_dict = model.state_dict()
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    transform = transforms.Compose([transforms.ToTensor()])
    device = torch.device(cfg.device)
    train_dataset = datasets.MNIST(
        root=cfg.data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_path, train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    model = layerwise_genetic_train(
        cfg.shape_list,
        train_loader,
        cfg.pop_size,
        cfg.generations,
        device,
        cfg.finetune_epoch,
        cfg.finetune_lr,
    )

    return model


if __name__ == "__main__":
    main()
