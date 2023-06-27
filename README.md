# [In-Progress] MusicSpectroGAN
Generative Adversarial Network to generate music via spectrogram image of music.

## Description :scroll:

## General Requirements :mage_man:
* If using GPU for training, atleast 4GB of VRAM is required for 64x64x1 image size and 64 batch size.

## Code Requirements :mage_woman:

## Dataset 💾

### Preprocess dataset
```python
python preprocess.py --input <path to input dataset directory> --output <path to save processed dataset>
```
<b>Arguments:</b><br/>
--input - Path to input dataset directory. The directory should contain class sub-directories with music files.<br/>
--output - Optional argument. Path to save processed dataset. Processed data is spectogram images.

### Training :running_man:
```python
python run.py --gan <model architecture>
```
<b>Arguments:</b><br/>
--gan - Specify GAN model architecture. Allowed values: dcgan, cgan.<br/>

## References :page_facing_up:
* PokeGAN: Generating Fake Pokemon with a Generative Adversarial Network | [Article](https://blog.jovian.com/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d)
* cGAN: Conditional Generative Adversarial Network — How to Gain Control Over GAN Outputs | [Article](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)

Happy Learning! 😄
