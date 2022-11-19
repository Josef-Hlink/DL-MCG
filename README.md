# MLP, CNN, GAN

Koen, Willem & Josef

Cats data source: [fferlito/Cat-faces-dataset](https://github.com/fferlito/Cat-faces-dataset).

Part 1 of the gzipped data (<100MB) is present in this repo.

Full file (cats.tar.gz) is hosted on [Josef's Google Drive](https://drive.google.com/file/d/1eJElIjkH8TIeEjANcOflliT9apRzXpWE/view?usp=sharing).


## Usage

Download the full cats.tar.gz file, put it in the data folder and extract.
Alternatively, just do the same with the cats-1.tar.gz file (no download needed).

```
mkdir data
cd data
wget https://drive.google.com/file/d/1eJElIjkH8TIeEjANcOflliT9apRzXpWE/view?usp=sharing
tar -xvzf cats.tar.gz
```

## Recommended structure of ignored dirs

```
data
├── cats       [d] ~30K png images
├── cats-1     [d] ~10K png images (sample of cats)
├── cats.npy   [f] array ~24k x 64 x 64 x 3
├── images.npy [f] array ~18k x 150 x 150 (clocks)
└── labels.npy [f] labels for images.npy
results
├── csv        [d] storage for dataframes
├── models     [d] storage for models (h5)
└── plots      [d] storage for plots (png)
```
