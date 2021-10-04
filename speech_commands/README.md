# README for KWS Speech Commands model training and evaluation

Please follow the instructions below to train and evaluate my keyword spotting models with the parameters and keyword set found in the `train.py` script. Parameter settings and keyword set can be configured as desired.

The average test accuracies I obtained for my trained ``cnn`` model and the baseline ``conv`` model both found in the `models.py` file across all evaluated keyword sets of 8 words are given in the `model_results` text file in the main directory.

**Training and test results:**

As mentioned, the training parameters and keywords used to train the models can be found in the `train.py` script. The models are trained to classify 8 keywords. This directory (`speech_commands`) needs to be downloaded and entered through a Linux terminal before proceeding with the following: 

1) To train my CNN model, ``cnn``, and obtain number it's parameter count and test accuracy,
 run the following in terminal:
```
python train.py --model_architecture='cnn'
```
2) To train my recurrent model, ``rnn``, and obtain it's parameter count and test accuracy,
 run the following in terminal:
```
python train.py --model_architecture='rnn'
```
- To train a quantized model for eight bit deployment, change the ``quantize`` flag in the train.py script to 'True'
- I quantized my CNN and RNN models for eight bit deployment after training and evaluated the quantized models.

**Freezing trained model:**

In order to obtain a frozen representation of a model using the trained checkpoint from the previous step, do as follows:

1) To obtain the frozen graph of the ``cnn`` model, run the following in terminal:
```
python freeze.py --model_architecture='cnn' --start_checkpoint='/tmp/speech_commands_train/cnn.ckpt-30000' --output_file='/tmp/cnn'
```
2) 1) To obtain the frozen graph of the ``rnn`` model, run the following in terminal:
```
python freeze.py --model_architecture='rnn' --start_checkpoint='/tmp/speech_commands_train/rnn.ckpt-30000' --output_file='/tmp/rnn'
```

**Test streaming accuracy:**

To test streaming performance of models on synthetically created audio by using the frozen models from the previous step,
run the following to generate the streaming audio file and its corresponding ground truth label file:
```
python generate_streaming_test_wav.py
```

1) To obtain steaming accuracy of the ``cnn`` model on the generated audio file, run the following in terminal:
```
python test_streaming_accuracy.py --wav='/tmp/speech_commands_train/streaming_test.wav'  --ground-truth='/tmp/speech_commands_train/streaming_test_labels.txt' --labels='labels.txt' --model='/tmp/cnn'
```
2) To obtain steaming accuracy of the `rnn` model on the generated audio file, run the following in terminal:
```
python test_streaming_accuracy.py --wav='/tmp/speech_commands_train/streaming_test.wav'  --ground-truth='/tmp/speech_commands_train/streaming_test_labels.txt' --labels='labels.txt' --model='/tmp/rnn'
```




