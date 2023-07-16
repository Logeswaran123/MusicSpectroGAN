# [In-Progress] 🎵 MusicSpectroGAN 🎵
Generate music via spectrogram image.

### FMA Dataset Directory structure
```
    fma
     |
     |- fma_small
     |      |- fma_small
     |           |- 000
     |           |- 001
     |           |- ...
     |- fma_metadata
     |      |- tracks.csv
```

### GTZAN Dataset Directory structure
```
    GTZAN
     |
     |- blues
     |- classical
     |- country
     |- ...
```

## Description :scroll:
Generative Adversarial Network (GAN) are used to generate new data samples that resemble a given training dataset.

The generator network takes random noise as input and generates synthetic data samples, such as images, audio, or text. Initially, the generator produces random outputs that do not resemble the real data. However, as the training progresses, the generator learns to generate samples that are increasingly similar to the real data.

The discriminator network, on the other hand, acts as a binary classifier. It takes both real data samples from the training set and generated samples from the generator as input. Its goal is to distinguish between the real and fake samples. The discriminator is trained to correctly identify the real data samples and classify the generated samples as fake.

During the training process, the generator and discriminator are trained in an adversarial manner. The generator aims to produce samples that can fool the discriminator into classifying them as real, while the discriminator tries to become more accurate in distinguishing between real and fake samples. This adversarial feedback loop between the two networks helps them improve over time.

Conditional Generative Adversarial Network (cGAN), extends the traditional GAN framework by incorporating conditional information. It allows the generator and discriminator to receive additional input, such as class labels or reference images, which guides the generation process. The conditional information helps control and influence the generated samples, allowing for targeted outputs.

Deep Convolutional Generative Adversarial Network (DCGAN) is a specific architecture of a GAN that employs convolutional neural networks (CNNs) for both the generator and discriminator. The use of CNNs in DCGANs enables hierarchical feature learning and helps produce high-quality, coherent images.

MusicSpectroGAN uses cGAN, DCGAN architectures to train a generator model with spectrogram images of music dataset. The model aims to generate coherent spectrogram images that can be converted to meaningful music. The model is conditioned on various music genres (classes).

## General Requirements :mage_man:
* If using GPU for training, atleast 4GB of VRAM is required for 64x64x1 image size and 64 batch size.

## Code Requirements :mage_woman:

## Dataset 💾
The project is tested with FMA and GTZAN datasets.

### Preprocess dataset
Convert music data into spectrogram images.
```python
python preprocess.py --dataset <gtzan or fma> --input <path to input dataset directory> --output <path to save processed dataset>
```
<b>Arguments:</b><br/>
--dataset - Choose the dataset to pre-process for training. Choices = ["gtzan", "fma"].<br/>
--input - Path to input dataset directory. The directory should contain class sub-directories with music files.<br/>
--output - Optional argument. Path to save processed dataset. Processed data is spectogram images.

## Training :running_man:
```python
python run.py --gan <model architecture> --dataset <path to spectrogram images dataset>
```
<b>Arguments:</b><br/>
--gan - Specify GAN model architecture. Allowed values: dcgan, cgan.<br/>
--dataset - Path to spectrogram images dataset. Obtained from the pre-processing step.<br/>

## References :page_facing_up:
* PokeGAN: Generating Fake Pokemon with a Generative Adversarial Network | [Article](https://blog.jovian.com/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d)
* cGAN: Conditional Generative Adversarial Network — How to Gain Control Over GAN Outputs | [Article](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)

Happy Learning! 😄
