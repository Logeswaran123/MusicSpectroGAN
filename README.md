# [In-Progress] MusicSpectroGAN
Generative Adversarial Network to generate music via spectrogram image of music.

## Description :scroll:

## General Requirements :mage_man:

## Code Requirements :mage_woman:

## Dataset 💾

## How to run :running_man:
For DCGAN, execute the following command,
```python
python dcgan.py
```

### Preprocess dataset
```python
python preprocess.py --input <path to input dataset directory> --output <path to save processed dataset>
```
<b>Arguments:</b><br/>
--input - Path to input dataset directory. The directory should contain class sub-directories with music files.<br/>
--output - Optional argument. Path to save processed dataset. Processed data is spectogram images.

### [In Progress] Training
```python
```

## References :page_facing_up:
* PokeGAN: Generating Fake Pokemon with a Generative Adversarial Network | [Article](https://blog.jovian.com/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d)
* cGAN: Conditional Generative Adversarial Network — How to Gain Control Over GAN Outputs | [Article](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)

Happy Learning! 😄
