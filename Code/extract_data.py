# Extracts data to json file from data set inputs
import json
import os
import math
import librosa

# Data set of sample songs for each genre
DATASET_PATH = "Genres"

# This is where data is saved to
JSON_PATH = "Data/sample_data.json"

# Basic sample rate
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # Length of sample song
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Defines function to save data to json file
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # Stores mapping, labels, and MFCCs as dictionaries
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Checks if genre folder is in path
        if dirpath is not dataset_path:

            # Save genre label to the mapping dictionary
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # Process all audio files
            for f in filenames:

		        # Loads audio files
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process all segments of audio files
                for d in range(num_segments):

                    # Calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # Extract MFCC
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # Store only MFCC feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # Save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
# Run function to save data to json file  
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=5)