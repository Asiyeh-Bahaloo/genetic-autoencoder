"""This script reads an image and classifies it."""
import os
import logging
import pickle
import torch
import cv2

from models import TiedAutoEncoder

from omegaconf import DictConfig
import hydra

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="../configs", config_name="inference_config")
def main(cfg: DictConfig):
    """This function reads an image and classifies it.

    Parameters
    ----------
    cfg : DictConfig
        config for inference
    """
    image = cv2.imread(cfg.data_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        # preprocess the image
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        image = torch.tensor(image, dtype=torch.float).unsqueeze(dim=0)
        # create model and load weights
        # unsuperwised model
        model = TiedAutoEncoder(cfg.shape_list, nonlinearity=torch.relu)
        model.load(folder_path=cfg.weights_folder)
        encoded_feats, reconstructed_output = model(image)
        # superwised model
        svm_path = os.path.join(cfg.weights_folder, "svm_weights.pickle")
        clf = pickle.load(open(svm_path, "rb"))
        log.info(f"Loaded SVM from {svm_path}")
        predict = clf.predict(encoded_feats)
        log.info(f"The predicted number is: {predict.squeeze()}")
        # draw the outputs
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(image.squeeze())
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.imshow(reconstructed_output.detach().numpy().reshape(28, 28))
        fig.savefig(os.path.join(cfg.output_dir, os.path.basename(cfg.data_path)))
        log.info(f"Reconstructed output wrote in: {cfg.output_dir}")
    else:
        print("Can't find the image for prediction")


if __name__ == "__main__":
    main()
