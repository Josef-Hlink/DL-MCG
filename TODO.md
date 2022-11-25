# TODO

## Task 1

- [ ] 1.1
    - [x] Play around with [this repo](https://github.com/keras-team/keras/tree/tf-keras-2/examples) (keras examples) some more
- [ ] 1.2
      [ ] Play around with dropouts, weight initializations, and running an MLP on a 3D dataset (RGB images)
    - [ ] Try to get CIFAR-10 to work with the MLP (maybe just with the single best model found by the F-MNIST MLP)
    - [ ] Very simple grid search over CNN architectures (number of layers, number of filters, kernel size)
    - [ ] Report on findings

## Task 2

- [ ] 2.1
    - [ ] Implement the pre-trained network found in the [helper notebook](Task2_helper.ipynb)
    - [ ] Simple regression
    - [ ] "Common sense" regression
    - [ ] Classification (24 classes, so for each half hour)
    - [ ] Multi-head model (similar to ?simple? regression)
- [ ] 2.2
    - [ ] Achieve a good accuracy on all tasks (common sense error < 10 min, so 0.17)
    - [ ] Report on findings

## Task 3

- [ ] 3.2
    - [ ] Find a way to pull and store the data from Josef's google drive so no setup is needed
- [ ] 3.3
    - [ ] Tinker with the architectures
    - [ ] Restructure code, so it's readable
- [ ] 3.4
    - [ ] Implement interpolation in VAE latent space
- [ ] 3.5
    - [ ] Report on all findings

## General

- [ ] Create setup.py instead of two separate requirements.txt files
- [ ] Remove fluff, so only the stuff that matters gets into the zip file
- [ ] Update README.md
