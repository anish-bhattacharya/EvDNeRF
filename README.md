# EvDNeRF: Reconstructing Event Data with Dynamic Neural Radiance Fields

This codebase is based on the paper EvDNeRF: Reconstructing Event Data with Dynamic Neural Radiance Fields.

## Installation

### Preferred method (docker)

We provide a docker container that can be pulled from dockerhub.
```
docker pull evdnerf/evdnerf:dev-user
```

```
docker run -d --rm -it --gpus all -v /home/$USER/evdnerf:/home/user/evdnerf evdnerf/evdnerf:dev-user
```

This will run a docker container image, the name of which can be seen in the output of `docker ps`. Start a shell in the docker container via:
```
docker exec -it -u root <running_docker_image_name> bash
```

Now you should be able to run an example train or test configuration (see following sections).

### Alternate method (conda env, borrowed from D-NeRF)

(Similar to [D-NeRF installation](https://github.com/albertpumarola/D-NeRF)).
From inside the evdnerf directory:
```
conda create -n evdnerf python=3.6
conda activate evdnerf
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ..
```

You will have to install additional dependencies manually; we have not provided an updated requirements.txt yet.

## Train

### Datasets

We provide datasets `public_datasets` at the following link to train EvDNeRF models from simulated and real datasets. After extracting the datasets tar, please read the contained, short README.

[Data Google Drive Link](https://drive.google.com/drive/folders/1fIMukIFCYYE7u_wk-tuFnlyE4mrfCA0i?usp=sharing)

### Example training experiment

To train from the `jet-down-32` dataset, we will use the `configs/jet_train.txt` config file. Edit the `root_path` and `datadir` to point to your `evdnerf` directory and inside the `public_datasets` directory, respectively.

You may add a `ft_file` line (refer to the test config file for an example) if you wish to continue training from a weights checkpoint. You may also change the lines `i_print` (prints training stats to console every `i_print` iterations), `i_img` (runs validation and saves event images ang grayscale images every `i_img` iterations), `testskip` (one of how many frames to render when doing validation), `i_weights` (saves weights every `i_weights` iterations), `N_iter` (how many iterations to train). We present the best results on simulated data, for example, by training for 100k iterations on 32 event frames (e.g., `jet-down-32` dataset), then 50k on 64 (e.g., `jet-down-64` dataset), and 50k on 128 (e.g., `jet-down-128` dataset). 100k on 32 and 100k on 128 also yields similar results.

Run the following command to start training:
```
python run_evdnerf.py --config configs/jet_train.txt
```
This command will run training on the `jet-*` dataset. Progressive weights and validation are saved to a log folder: `./logs/<exp_datetime>`. Any weights file absolute path can then be inputted as the `ft_file` argument to test on.

To train a normal D-NeRF model without events, refer to `jet_train_dnerf.txt`; the key argument is `dnerf`.

## Test

### Pre-trained weights

First download pre-trained weights. They should placed in the directory `evdnerf/pretrained_weights`. This is the same link as the one above.

[Data Google Drive Link](https://drive.google.com/drive/folders/1fIMukIFCYYE7u_wk-tuFnlyE4mrfCA0i?usp=sharing)

### Example test experiment

There are various testing configurations, including event frame generation directly from `json` files specifying poses and timestamps (via setting `render_test_path`), or calculating and comparing metrics against an established test set (via setting `render_test`).

For an example, we set `render_test_path` to render predicted events for any defined `json` file. Various such files are found in `test_configs/`. `test_configs/transforms_test_validation.json` contains validation viewpoints each with 32 timesteps. To test a model trained on the `jet-down` dataset, we will use the `configs/jet_test.txt` config file, where `ft_file` points to the corresponding pretrained weights file. Edit the `root_path` and `datadir` in the config file to point to your `evdnerf` directory and inside the `public_datasets` directory, respectively.

To test the model, run:
```
python run_evdnerf.py --config configs/jet_test.txt
```
When finished, resulting event batches in the form of images are found in `./logs/<exp_datetime>/<exp_datetime>/evim/*.png`.

## Data generation

Coming soon!
