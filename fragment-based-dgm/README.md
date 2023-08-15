## Code for the Dissertation "A Deep Fragment-Based Generative Model for De Novo Molecular Design"
### Report
Link to [Research paper and findings](https://drive.google.com/file/d/1iilahcVytCHjJU3EMK2Gi9Ytd-LH7rBs/view?usp=sharing)


### This code is adapted from the publicly available code by Podda et al. (2020) for the paper "A Deep Generative Model for Fragment-Based Molecule Generation" (AISTATS 2020)
### The original paper and the proceedings can be found through the following links:
Links:  [Paper](http://proceedings.mlr.press/v108/podda20a/podda20a.pdf) - [AISTATS 2020 proceedings](http://proceedings.mlr.press/v108/)

Github Repository: [link](https://github.com/marcopodda/fragment-based-dgm)
### Data and Models
Please download the data from the following [link](https://drive.google.com/drive/folders/1rvyRWvwjaRs3b-rXx4AZBJPP4L1GE4Ge?usp=sharing) and our model, the benchmark model and the reference model from the following [link](https://drive.google.com/drive/folders/1puL7k0dikxZT3pfrPr1zSsriei-x0ew6?usp=sharing) 

### Cloning this repository
Once this repository is cloned, it is vital to download the entire folder from both links (the 'DATA' folder and the 'RUNS' folder) and unzipping them and only keeping the 'DATA' and 'RUNS' folder in the same location as the cloned repository.

### Installation

Run:

`conda install -n <env_name> requirements.txt`

This will take care of installing all required dependencies.

### Training the model
To reproduce the training of our model, run the following:

`python manage.py train --dataset CHEMBL --use_gpu --no_mask --batch_size 16 --embed_size 100 --num_epochs 20 --hidden_layers 1 --hidden_size 100`

If you do not have a gpu omit the `--use_gpu` option.

Other options that start with `--` detail the model hyperparameters. To review all the hyperparameters that can be changed, please view the `utils/parser.py` or check out `python manage.py train --help`. 
It is however, vital to keep the hidden size hyperparameter as 100 respectively because the pre-trained model used to train our model has a size of 100

Training the model will create folder `RUNS` with the following structure:

```
RUNS
└── <date>@<time>-<hostname>-<dataset>
    ├── ckpt
    │   ├── best_loss.pt
    │   ├── best_valid.pt
    │   └── last.pt
    ├── config
    │   ├── config.pkl
    │   ├── emb_<embedding_dim>.dat
    │   ├── params.json
    │   └── vocab.pkl
    ├── results
    │   ├── performance
    │   │   ├── loss.csv
    │   │   └── scores.csv
    │   └── samples
    └── tb
        └── events.out.tfevents.<tensorboard_id>.<hostname>
```


the `<date>@<time>-<hostname>-<dataset>` folder is a snapshot of your experiment, which will contain all the data collected during training.

### Reproducing the plots and results from our model
open jupyter notebook, specifically the `Optimisation.ipynb` file and follow the step-by-step process.

This includes:
1. Running the training set through the model for latent space and box plots
2. Plotting latent space PCA for the training set
3. Sample valid molecules and plots of the latent space alongside box plots of sampled molecules
4. Sample and keep all points from the prior including plots of the latent space
5. Moving in the latent space
6. Checking the reconstruction of the training data

### Training the original model
To train the original model run the following line after cloning the [**Encoding_original** branch](https://github.com/panukorn17/drug_discovery_models/tree/Encoding_original/fragment-based-dgm):

`python manage.py train --dataset CHEMBL --use_gpu --no_mask --batch_size 16 --embed_size 100 --num_epochs 20 --hidden_layers 1 --hidden_size 100`

### Training the reference model
To train the reference model, please uncomment the beta list line 267 in the `learner/trainer.py` file and comment the beta list line 265
