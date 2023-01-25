import os
import logging
import pickle
from sklearn import svm
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from omegaconf import DictConfig
import hydra

import matplotlib.pyplot as plt

# from genetic_autoencoder.core.models import *
from models import TiedAutoEncoder, GeneticTiedAutoEncoder

log = logging.getLogger(__name__)


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
            log.info(
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
        _, pred = model(data)
        target = torch.flatten(data, start_dim=1)
        loss = F.mse_loss(pred, target)
        val_loss += loss

    validation_loss = val_loss / len(data_loader)
    return validation_loss


@torch.no_grad()
def evaluate_classification_model(model, clf, data_loader, device):
    """Evaluate the model and calculate the MSE loss for classification.

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
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        feat, _ = model(data)
        predict = clf.predict(feat)
        loss = F.mse_loss(
            torch.tensor(predict, dtype=torch.float),
            target.type(torch.float),
        )
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
        log.info(f"training .... \n {model}")
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
    log.info("The whole autoencoder has been trained using SGD algorithm")


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
    log.info(f"\nGenerated {pop_size} models with {shape_list} layer size! ")
    models = [
        GeneticTiedAutoEncoder(shape_list, nonlinearity=torch.relu)
        for i in range(pop_size)
    ]

    # Load prevoius layer weights
    if prev_weights is not None:
        for model in models:
            model.load_state_dict(prev_weights, strict=False)
        shapes = []
        for key_weight in prev_weights:
            shapes.append(prev_weights[key_weight].shape)
        log.info(f"\nLoaded prevoius layer weights with shape {shapes}")

    layer_key = next(reversed(models[0].state_dict()))  # last layer key
    log.info(
        f"\nTraining {layer_key} with shape of {models[0].state_dict()[layer_key].shape} : "
    )

    for g in range(generations):
        log.info(f"\nGeneration {g + 1}: \n")
        log.info("Calculating fitness for each chromosome...")
        # fitness
        fitness = []
        for model in models:
            fitness.append(1 / evaluate_loss(model, train_loader, device))
        log.info(f"The fitness values in begining: {fitness}")
        # Select 5 best models (/chromosomes)
        fit_arg = np.argsort(fitness)
        selected_models = [models[f] for f in fit_arg[: pop_size // 2]]
        log.info(f"Selected {pop_size // 2} Top best chromosomes")

        # finetune selected models
        log.info("Finetuning selected models...\n")
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
        log.info(
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
    log.info("Selecting best chromosome as answer...")
    # fitness
    for model in models:
        fitness.append(1 / evaluate_loss(model, train_loader, device))
    fit_arg = np.argsort(fitness)
    log.info(f"The fitness values of last generation: {fitness}")
    best_model = models[fit_arg[-1]]
    log.info(f"The total loss for the train dataset is: {1/fitness[fit_arg[-1]]}")
    log.info(f"Finished the training for chromosome {layer_key}")
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
    log.info("The whole autoencoder has been trained using genetic algorithm")
    return model


def draw_sample_output(model, data_sample, path):
    """draw and save output reconstructed image.

    Parameters
    ----------
    model : torch.model
        The model for calculating the reconstruction.
    data_sample : np.array
        The unsqueezed image.
    path : str
        Path for saving the output image.
    """
    _, reconstructed_output = model(data_sample)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(data_sample.squeeze())
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(reconstructed_output.detach().numpy().reshape(28, 28))
    fig.savefig(path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # create data loaders
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
    # train the autoencoder using genetic or sgd
    if cfg.method == "genetic":
        model = layerwise_genetic_train(
            cfg.shape_list,
            train_loader,
            cfg.pop_size,
            cfg.generations,
            device,
            cfg.finetune_epoch,
            cfg.finetune_lr,
        )
        # model = GeneticTiedAutoEncoder(cfg.shape_list, nonlinearity=torch.relu)
    elif cfg.method == "sgd":
        model = TiedAutoEncoder(cfg.shape_list, nonlinearity=torch.relu)
        layer_wise_train(
            model, device, train_loader, cfg.lr, cfg.epochs, loss_fn=F.mse_loss
        )
    # save the model
    out_path = os.path.join(cfg.output_dir, cfg.model_name)
    # create the dir if not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    model.save(out_path)

    # unsuperwised model validation
    data_sample = train_dataset[2][0]
    draw_sample_output(model, data_sample, os.path.join(out_path, "recons_img.png"))

    # evaluate the unsuperwised model for reconstruction
    if cfg.evaluate:
        log.info(
            f"The total loss for the test dataset is: {evaluate_loss(model, test_loader, device)}"
        )

    # train the classification head (svm)
    all_encoded_feats = []
    all_targets = []
    clf = svm.SVC(gamma="scale", kernel="rbf")

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        encoded_feats, _ = model(data)
        all_encoded_feats.append(encoded_feats)
        all_targets.append(target)

    all_encoded_feats = np.concatenate(all_encoded_feats, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    clf.fit(all_encoded_feats[:10], all_targets[:10])
    # evaluate the classification head on train set
    predicts = clf.predict(all_encoded_feats)
    print("predicts", predicts)
    print("all_targets", all_targets)
    loss = F.mse_loss(
        torch.tensor(predicts, dtype=torch.float),
        torch.tensor(all_targets, dtype=torch.float),
    )
    log.info(f"The total classification loss for the train dataset is: {loss}")

    # save the superwised model
    pickle.dump(clf, open(os.path.join(out_path, "svm_weights.pickle"), "wb"))
    log.info("Saved SVM Sucessfully!")

    # evaluate the classification head on test set
    if cfg.evaluate:
        classification_loss = evaluate_classification_model(
            model, clf, test_loader, device
        )
        log.info(
            f"The total classification loss for the test dataset is: {classification_loss}"
        )
    return model


if __name__ == "__main__":
    main()
