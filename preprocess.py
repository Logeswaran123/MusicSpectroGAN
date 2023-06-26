import argparse
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str,
                                        help="Path to input dataset directory.")
    parser.add_argument('-o', "--output", required=False, type=str,
                                        help="Path to save processed dataset.")
    return parser


def create_mel_spectogram(input_dir_path: str, output_dir_path: str = None):
    """
    Create mel spectogram for audio files.

    Arguments:
    input_dir_path: Path to dataset directory with labels directories containing audio files.
    output_dir_path: Path to save processed dataset (mel spectograms).
    """
    if output_dir_path is None:
        output_dir_path = os.getcwd()

    output_path = output_dir_path + "/spectrogram_images"
    if not os.path.exists(output_path):
        print(f"Creating processed dataset directory: {output_path}\n")
        os.makedirs(output_path)

    for dir in os.listdir(input_dir_path):
        label = str(dir)
        label_dir_path = input_dir_path + "/" + label

        print(f"Processing {label} dataset in path: {label_dir_path}...\n")

        for file_name in os.listdir(label_dir_path):
            out_label_sgram_path = output_path + "/" + label
            if not os.path.exists(out_label_sgram_path):
                print(f"Creating {label} processed dataset directory: {out_label_sgram_path}\n")
                os.makedirs(out_label_sgram_path) 

            file_path = label_dir_path + "/" + file_name
            try:
                samples, sample_rate = librosa.load(file_path)
            except:
                print(f"Failed to process {file_path}\n")
                continue
            sgram = librosa.stft(samples)
            sgram_mag, _ = librosa.magphase(sgram)
            mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
            mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

            plt.axis('off')
            plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel_sgram, cmap='gray_r')
            save_file_name = out_label_sgram_path + "/" + os.path.splitext(file_name)[0] + ".png"
            plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)
            plt.close()


if __name__ == "__main__":
    args = argparser().parse_args()
    create_mel_spectogram(args.input, args.output)