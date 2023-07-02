import argparse
import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, type=str, choices=["gtzan", "fma"],
                                        help="GTZAN or FMA dataset.")
    parser.add_argument('-i', "--input", required=True, type=str,
                                        help="Path to input dataset directory.")
    parser.add_argument('-o', "--output", required=False, type=str,
                                        help="Path to save processed dataset.")
    return parser


# Source: https://gist.github.com/drscotthawley/eb4ffb1ec4de29632403c1db396e419a
def fma_process(fma_small_path, tracks_csv_path):
    import ast
    import pandas as pd
    import soundfile as sf

    def get_audio_path(audio_dir, track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

    if 'tracks' in tracks_csv_path:
        tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

    processed_fma_path = os.getcwd() + '/samples'
    os.mkdir(processed_fma_path)

    small = tracks['set', 'subset'] <= 'small'
    y_small = tracks.loc[small, ('track', 'genre_top')]
    sr = 44100
    for track_id, genre in y_small.iteritems():
        if not os.path.exists(processed_fma_path + '/' + genre):
            os.mkdir(processed_fma_path + '/' + genre)

        mp3_filename = get_audio_path(fma_small_path, track_id)
        out_wav_filename = processed_fma_path + '/' + genre + '/' + str(track_id) + '.wav'
        in_wav_filename = out_wav_filename
        cmd = 'ffmpeg -hide_banner -loglevel panic -i ' + mp3_filename + ' ' + in_wav_filename
        print("--------\nExecuting conversion: " + cmd)
        os.system(cmd)

        print("Reading ", in_wav_filename)
        try:
            data, sr = librosa.load(in_wav_filename, sr=sr, mono=True)
        except:
            print(f"Failed to laod {in_wav_filename}\n")
            continue

        print("Writing ", out_wav_filename)
        # librosa.output.write_wav(out_wav_filename, data, sr=sr)
        sf.write(out_wav_filename, data=data, samplerate=sr)


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

    if args.dataset == "fma":
        # FMA Dataset Directory structure
        #   fma
        #    |
        #    |- fma_small
        #    |      |- fma_small
        #    |           |- 000
        #    |           |- 001
        #    |           |- ...
        #    |- fma_metadata
        #    |      |- tracks.csv
        #
        # Pass path to /fma directory.
        fma_process(args.input + "/fma_small/fma_small/", args.input + "/fma_metadata/tracks.csv")
        wav_samples_path = os.getcwd() + '/samples'
        create_mel_spectogram(wav_samples_path, args.output)
        shutil.rmtree(wav_samples_path)
    elif args.dataset == "gtzan":
        # GTZAN Dataset Directory structure
        #   GTZAN
        #    |
        #    |- blues
        #    |- classical
        #    |- country
        #    |- ...
        #
        # Pass path to /GTZAN directory.
        create_mel_spectogram(args.input , args.output)