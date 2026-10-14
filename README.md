# Attribute-Controlled Image Manipulation using Diffusion Models

<img width="1567" height="337" alt="Screenshot 2026-10-13 130607" src="https://github.com/user-attachments/assets/a8b108cb-3118-4834-9b66-47383b249667" />

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

`W_final_edit = W_edited - W_original` (w_final_edit is further normalized to 0 mean)
### Dataset Generation
| **Original Face** | **Face Pair with Bob Cut Attribute** |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/11f49cf6-4f99-4130-9a4d-f87a2b508739" width="350"/> | <img src="https://github.com/user-attachments/assets/69b05982-3c9b-4b84-8f34-9627fef323a8" width="350"/> |

## Model Architecture Overview

The denoising network uses fully connected layers to predict the noise added to a flattened edit direction vector of size 9216 (18 × 512).

A 128-dimensional sinusoidal timestep embedding is processed through an MLP and injected into 8 residual blocks with 2048 hidden neurons each. Every block uses LayerNorm and SiLU activations along with linear layers and residual connections.

The final layers predict the noise during the forward pass. A timestep-dependent skip connection is also used in the final output to help the network learn corrections to the input.

## Training Process

The model is trained to predict the noise added to latent edit direction vectors.

1. **Load Data:** Load batches of latent edit direction vectors from the custom dataset.
2. **Sample Noise:** Generate random Gaussian noise and randomly select a timestep for each sample.
3. **Forward Diffusion:** Add noise to the original edit directions using the noise scheduler according to the sampled timesteps.
4. **Noise Prediction:** Pass the noisy vectors and timesteps through the MLP denoising network to predict the added noise.
5. **Calculate Loss:** Compute Mean Squared Error (MSE) between the predicted noise and the actual noise.
6. **Update Weights:** Backpropagate the loss and update the model parameters using AdamW. Gradient clipping is applied to stabilize training.
7. **Save Model:** Save the model weights for checkpoints.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | Fully Connected MLP |
| Input Dimension | 9216 (18 × 512) |
| Hidden Dimension | 2048 |
| Timestep Embedding Dimension | 128 |
| Number of Residual Blocks | 8 |
| Diffusion Timesteps | 1000 |
| Noise Scheduler | Linear |
| Beta Start | 0.0001 |
| Beta End | 0.02 |
| Batch Size | 128 |
| Number of Epochs | 100 |
| Optimizer | AdamW |
| Learning Rate | 0.0002 |
| Weight Decay | 0.01 |
| Loss Function | Mean Squared Error (MSE) |

The model learns to predict the Gaussian noise added to each edit direction vector at a randomly selected diffusion timestep.

## Editing Process

A noisy flattened vector of shape `(1, 18 × 512)` is sampled and passed through the reverse diffusion process over 1000 timesteps to generate a coarse variation edit direction vector.

this edit direction vector is scaled by a parameter gamma(γ) before adding the mean back this helps control the diversity of the images
then the final edit latent vector is scaled by lambda(λ) to determine strength of the edit and then added to the encoded latent vector of the input image through e4e(encoder4editing or any test latent vector of a face) then finally passed through the decoder to generate the edited face. 

To generate fine variations the output at timestep \(t_0\) is saved and used as the starting point for further reverse diffusion runs. Repeating this process \(n\) times produces \(n\) fine variation latent vectors which are then processed in a similar way as before and passed through the StyleGAN2 decoder to generate the corresponding attribute edits.

The lower the selected timestep the finer the variations produced.

### Editing Process

<img src="https://github.com/user-attachments/assets/5a740bec-2a7d-428f-98bc-fa1507dbf7e5" width="100%"/>

## Outputs

### Hairstyle

| Output 1 | Output 2 |
|:---:|:---:|
| <img width="3385" height="1014" alt="image" src="https://github.com/user-attachments/assets/c61b8a52-3ab3-4e6b-a6eb-616facb8bf1d" />| <img width="3385" height="1014" alt="Screenshot 2026-10-13 235009" src="https://github.com/user-attachments/assets/00fe25cf-00aa-4056-9742-565209db72c3" />|

### Eyeglasses

| Output 1 | Output 2 |
|:---:|:---:|
| <img width="1597" height="376" alt="Screenshot 2026-10-13 015200" src="https://github.com/user-attachments/assets/e0bd3c9e-f99e-423b-a1be-314d1b798ff6" />|<img width="1608" height="357" alt="Screenshot 2026-10-13 150104" src="https://github.com/user-attachments/assets/6b3f0a6d-918a-46b6-accc-996b9a185881" />|
### Smile

| Output 1 | Output 2 |
|:---:|:---:|
|<img width="1572" height="377" alt="Screenshot 2026-10-13 154404" src="https://github.com/user-attachments/assets/02493267-c3e3-4984-ba7b-44ec9151ea72" />| <img width="1516" height="333" alt="Screenshot 2026-10-13 010610" src="https://github.com/user-attachments/assets/90b03ec5-20af-4bea-98f4-cbb63890fe23" />|

### Age

| Output 1 | Output 2 |
|:---:|:---:|
| <img width="1563" height="337" alt="Screenshot 2026-10-13 201416" src="https://github.com/user-attachments/assets/36c98a75-3100-47d4-8b46-1ea35f9d85af" /> |<img width="1562" height="325" alt="Screenshot 2026-10-13 001603" src="https://github.com/user-attachments/assets/1d78ed1a-35a8-48af-a204-7000f5d26539" />|
