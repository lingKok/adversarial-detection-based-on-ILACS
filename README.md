# Adversarial detection based on ILACS


## 1. Description
Adversarial detection based on ILACS is an adversarial detection method, test our method on an open platform DEEPSEC.

![system](./framework.png)


### 1.1 Glance at the DEEPSEC Repo:

- `RawModels/` contains code used to train models and trained models will be attacked;
- `CleanDatasets/` contains code used to randomly select clean samples to be attacked;
- `Attacks/` contains the implementations of attack algorithms.
- `AdversarialExampleDatasets/` are collections of adversarial examples that are generated by all kinds of attacks;
- `Defenses/` contains the implementations of defense algorithms;
- `DefenseEnhancedModels/` are collections of defense-enhanced models (re-trained models);
- `Evaluations/` contains  code used to evaluate the utility of attacks/defenses and the security performance between attack and defenses.

### 1.2 Requirements:

Make sure you have installed all of following packages or libraries (including dependencies if necessary) in you machine:

1. PyTorch 1.9.0
2. TorchVision 0.10
3. numpy, scipy, PIL, skimage, tqdm ...
4. [Guetzli](https://github.com/google/guetzli): A Perceptual JPEG Compression Tool

### 1.3 Datasets:
We mainly employ three benchmark dataset MNIST, SVHN and CIFAR10.


## 2. Usage/Experiments


### STEP 1. Training the raw models and preparing the clean samples
We firstly train and save the deep learning models for MNIST, SVHN and CIFAR10 [here](./RawModels/), and then randomly select and save the clean samples that will be attacked [here](./CleanDatasets/).

### STEP 2. Generating adversarial examples
Taking the trained models and the clean samples as the input, we can generate corresponding adversarial examples for each kinds of adversarial attacks that we have implemented in [`./Attacks/`](./Attacks/), and we save these adversarial examples [here](./AdversarialExampleDatasets/).

### STEP : Detecting
if you want to perform detection operation, you need to generate corresponding adversarial examples and then you can perform the detection operation, for examples:
python detect.py --dataset MNIST --attack FGSM --test_attack FGSM


## 3. Update
If you want to contribute any new algorithms or updated implementation of existing algorithms, please also let us know.
> UPDATE: All codes have been re-constructed for better readability and adaptability.
## 4.

