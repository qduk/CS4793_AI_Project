# CS4793_AI_Project
Genre classifying project for CS4793
CREATED_BY: Adam Byczkowski, Morgan Houston, Jonathan McDonald


## Python Libraries
We are using the following libraries to help us in analyzing music and build a neural network. 
* [TensorFlow](https://www.tensorflow.org/)
* [Librosa](https://github.com/librosa/librosa)

## Dataset for Training Neural Network
The GTZAN data for music genre classification can be found [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification). This contains 1.6 Gb of 30 second clips of all sorts of genres. This is awesome as we can use it to train our model. 


## audio_data.py
This python file was as a learning process to figure out how exactly we needed to extract data from a .wav file. We were able to learn to use fourier transforms and short term fourtier transforms to extract data from the .wav files. Once we were able to do it with one file, we were able to move on to pulling data about from a huge set of songs so that we could train our mode. 

## extract_data.py 
In order for us to be able to build a neural network to interpret songs, we had to research on how audio files are converted for music analysis. After some research we found that we needed to use MFCC's (Mel-frequency Cepstral Coefficients). In short, these coefficients are used to determine what typs of instruments are in a recording. This python file goes through each song in the dataset and creates MFCC's for them. It also maps those MFCC's generated to whichever specific genre they are. (Ex: blues = 0, classical = 1, etc ) 

## Tensorflow_Test1.py
This python file takes the json file created from "extract_data.py" and uses to it train and test our model. When we create the model, we are able to tell tensorflow how much of the dataset to use for training and how much to use for testing. Currently it uses 70% of the dataset for training and 30% for testing.

## Tensorflow_Test2.py
This python file is the same as Tensorflow_Test1.py up until the very end. The last few lines has the code choose a song at random and have the AI guess what genre it is. It then tells us what was guessed and what the actual genre of the song is. 

## Next Steps
Next we need to find a way to pass one song to our model. We are going to have it process that song and then return what genre it thinks it is. 