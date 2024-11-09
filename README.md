# Brain MRI Super-Resolution using Diffusion Models with Multi-Noise Scheduling

The baseline code is forked from [Image Super-Resolution via Iterative Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), which is based on the SR3 architecture.

## Implementation Details

This project extends the baseline code to specifically address Brain MRI Super-Resolution, incorporating MRI-specific noise handling techniques. Key modifications include:

### 1. Multi-Noise Scheduling
   - We introduced a **multi-noise scheduling framework** in the forward diffusion process to better model MRI-specific noise artifacts:
     - **Poisson Noise**: Applied in the initial steps to model intensity-dependent fluctuations.
     - **Rician Noise**: Added in the middle stages to capture non-Gaussian characteristics common in MRI data.
     - **Gaussian Noise**: Introduced in the final steps for standard noise handling.
   - The reverse diffusion process then progressively denoises each type in reverse order, helping the model effectively differentiate and remove each noise type.

### 2. Attention Mechanism Integration
   - To improve feature extraction and enhance detail preservation, an **Attention Module** was integrated into the UNet backbone. This attention mechanism helps the model focus on critical areas, which is essential for Brain MRI where fine details are often diagnostically relevant.

### 3. Extended Diffusion Steps
   - The number of diffusion steps was increased from 2000 to 3500, allowing sufficient intervals for each noise type in the multi-noise scheduling framework. This gives the model more time to learn distinct denoising strategies for Poisson, Rician, and Gaussian noise.

### 4. Enhanced Loss Function
   - The loss function was adapted to include **L1 and perceptual losses** that help retain fine structural details, essential for high-fidelity MRI outputs. This allows the model to achieve enhanced detail and edge preservation in super-resolved images.

### 5. New Configuration Options
   - The configuration files have been updated to support multi-noise scheduling and attention module parameters. Users can easily adjust settings for noise intensity ranges, intervals, and attention to customize for various MRI datasets.

## Summary of Modifications
These modifications improve the baseline model's ability to handle the unique noise profiles and high-resolution demands of Brain MRI data, resulting in state-of-the-art MRI Super-Resolution performance.


## Acknowledgements

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

Furthermore, we are benefitting a lot from the following projects:
- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hejingwenhejingwen/AdaFM
