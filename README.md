# simple-gcm-deep-learning
code developed for the paper "Toward Data‐Driven Weather and Climate Forecasting: Approximating a Simple General Circulation Model With Deep Learning" (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GL080704)

The aim is to train a deep convolutional network on a run of a simplified general circulation model (climate model), using Keras and Tensorflow.
A sample of the climate model data can be found in the accompanying zenodo repository (10.5281/zenodo.1472023)


puma_CNN_preprocess_inputdata.py processed the raw climate model output data into a form suitable for the training

puma_CNN_tune_network.py  tries different neural network architectures and chooses the one that works best

puma_CNN_train_and_predict.py  does the training with the best architecture, and predicts on the test set

puma_CNN__analyze_data.py   analyses the predictions by the network

puma_CNN_make_climate_sims.py  uses the trained network to make a "climate"-run with successive network predictions 

largescale-ML.yml is a dump of the anaconda environment used.
