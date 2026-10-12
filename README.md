# Attribute-Controlled Image Manipulation using Diffusion Models

<img width="1616" height="357" alt="Screenshot 2026-10-11 185931" src="https://github.com/user-attachments/assets/3e1cfecb-ac2c-4d45-a0bc-dc48b4b76dd4" />

## Overview

The aim of this project is to develop an attribute-guided facial image manipulation system using **Denoising Diffusion Probabilistic Models (DDPMs)** to modify specific facial attributes while maintaining the original identity and visual realism of the face.

Given a source face and a target attribute the model generates a new image of the same face with the selected attribute edited while preserving identity and realism.

## Learning Phase

The project was built progressively to develop a strong conceptual and practical understanding.

1. Studied the fundamentals of Neural Networks.
2. Implemented MNIST digit classification using Artificial Neural Networks from scratch with NumPy.
3. Learned optimizers and hyperparameter tuning along with CNN fundamentals.
4. Implemented ANN and CNN models on MNIST using PyTorch.
5. Implemented a CNN on CIFAR-10 using different normalization and regularization techniques:
   - Batch Normalization
   - Dropout Regularization
6. Learned basic Probability Theory.
7. Learned the fundamentals of Generative AI by following the IITM *Mathematical Foundations of Generative AI* playlist.
8. Studied GANs and StyleGANs and implemented them on CIFAR-10.
9. Studied Denoising Autoencoders,Variational Autoencoders and Hierarchical Variational Autoencoder and implemented them on MNIST.
10. Learned DDPMs and studied the research paper before implementing DDPM on MNIST.
11. Finally studied Attribute Diffusion and implemented the approach after reading its research paper.

**Research Paper:** [Attribute Diffusion: Diffusion-Driven Diverse Attribute Editing (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Parihar_Attribute_Diffusion_Diffusion_Driven_Diverse_Attribute_Editing_WACV_2025_paper.pdf)

## Custom Dataset - Edit Direction Vectors

Since there is no readily available dataset containing pairs of faces with and without specific attributes we created our own dataset using existing editing tools. StyleCLIP was used to generate hairstyle variations along with other attributes such as eyeglasses and age and smile.

The original and edited images were passed through **Encoder4Editing (e4e)** for StyleGAN2 to obtain their latent vectors in \(W+) space. The difference between these vectors gives the final edit direction vector.

**Edit Direction Vector:**

`W_final_edit = W_edited - W_original`

### Dataset Generation

| **Original Face** | **Face Pair with Bob Cut Attribute** |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/11f49cf6-4f99-4130-9a4d-f87a2b508739" width="350"/> | <img src="https://github.com/user-attachments/assets/69b05982-3c9b-4b84-8f34-9627fef323a8" width="350"/> |

## Model Architecture Overview

The denoising network uses fully connected layers to predict the noise added to a flattened edit direction vector of size 9216 (18 × 512).

A 128-dimensional sinusoidal timestep embedding is processed through an MLP and injected into 8 residual blocks with 1024 hidden neurons each. Every block uses LayerNorm and SiLU activations along with linear layers and residual connections.

The final layers predict the noise during the forward pass. A timestep-dependent skip connection is also used in the final output to help the network learn corrections to the input.

## Training Process

The model is trained to predict the noise added to latent edit direction vectors.

1. **Load Data:** Load batches of latent edit direction vectors from the custom dataset.
2. **Sample Noise:** Generate random Gaussian noise and randomly select a timestep for each sample.
3. **Forward Diffusion:** Add noise to the original edit directions using the noise scheduler according to the sampled timesteps.
4. **Noise Prediction:** Pass the noisy vectors and timesteps through the MLP denoising network to predict the added noise.
5. **Calculate Loss:** Compute Mean Squared Error (MSE) between the predicted noise and the actual noise.
6. **Update Weights:** Backpropagate the loss and update the model parameters using AdamW. Gradient clipping is applied to stabilize training.
7. **Save Model:** Save the model weights after every epoch.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | Fully Connected MLP |
| Input Dimension | 9216 (18 × 512) |
| Hidden Dimension | 1024 |
| Timestep Embedding Dimension | 128 |
| Number of Residual Blocks | 8 |
| Diffusion Timesteps | 1000 |
| Noise Scheduler | Linear |
| Beta Start | 0.0001 |
| Beta End | 0.02 |
| Batch Size | 128 |
| Number of Epochs | 20 |
| Optimizer | AdamW |
| Learning Rate | 0.0002 |
| Weight Decay | 0.01 |
| Loss Function | Mean Squared Error (MSE) |

The model learns to predict the Gaussian noise added to each edit direction vector at a randomly selected diffusion timestep.

## Editing Process

A noisy flattened vector of shape `(1, 18 × 512)` is sampled and passed through the reverse diffusion process over 1000 timesteps to generate a coarse variation latent vector.

The generated latent vector is then passed through the StyleGAN2 decoder to obtain the edited face.

To generate fine variations the output at timestep \(t_0\) is saved and used as the starting point for further reverse diffusion runs. Repeating this process \(n\) times produces \(n\) fine variation latent vectors which are then passed through the StyleGAN2 decoder to generate the corresponding facial variations.

The lower the selected timestep the finer the variations produced. For experimental purposes we selected timestep **500**.

### Editing Pipeline

<img src="https://github.com/user-attachments/assets/5a740bec-2a7d-428f-98bc-fa1507dbf7e5" width="100%"/>

## Outputs

### Hairstyle

| Output 1 | Output 2 |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/a95c11b2-b829-42f1-9571-c49c7d6add1e" width="100%"/> | <img src="https://github.com/user-attachments/assets/7ddca587-80cd-41b2-b89b-528dafbc4079" width="100%"/> |

### Eyeglasses

| Output 1 | Output 2 |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/bdd1d899-9ed1-4a68-934c-936814ffb881" width="100%"/> | <img src="https://github.com/user-attachments/assets/079be5a9-7ded-48ac-a56d-e485f2f9510b" width="100%"/> |
