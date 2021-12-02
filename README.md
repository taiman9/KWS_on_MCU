# Deep Learning for keyword spotting (kws) on the edge:
Deep Learning project to train and implement keyword spotting models for speech command inference on the edge. I used Tensorflow's open source speech command recognition framework to develop and configure code for implementing my keyword spotting models for edge deployment. The objective of this project is to develop reliable keyword spotting models within the resource constraints of edge devices. Please view the *Report.pdf* file to find details about the keyword spotting pipelines I implemented.

The *keyword_recogntion_cnn* colab notebook is an end-to-end implementation for extracting features, training and evaluating my CNN model on a keyword set of 8 words. The *model_result* text file compares my cnn model's average test accuracy against that of a baseline cnn model (provided as ``conv`` in models.py) with about twice as many weight parameters as my cnn model across all keyword sets of 8 words I evaluated, and includes my cnn model's size. As can be seen in the text file, there is a difference of 3.5% in average test accuracy between the two models despite their vastly different sizes, given my model is developed to optimize accuracy within the resource contraints of edge platforms. It should be noted that I evaluated keyword sets with high percentages of similarly pronounced words and long words to test model robustness, both of which make classification harder and are not usually applicable in practical cases. 

The *speech_commands* directory contains the scripts and instructions to train my CNN and RNN models and obtain their test and streaming data results. The models are configured to train and evaluate the 8 keywords included in the file *labels.txt* in this directory. The number and selection of keywords on which models are trained and tested can be changed in the `train.py` script in the *speech_commands* directory, given the keywords belong to the Google speech command dataset used. The `models.py` script in the *speech_commands* directory and this directory contain my CNN model labelled as ``cnn``, my RNN model labelled ``rnn`` as well as default models provided for keyword spotting which are not designed/optimized for my keyword spotting use case.   
