# README for KWS Speech Commands model training and evaluation

Please follow the instructions below to train and evaluate my keyword spotting model against the basline model with the given parameter settings and keyword set. Parameter congifurations and keyword set can be changed if desired.

The results I obtained from my trained ``cnn`` model against the baseline ``conv`` model are contained 
in the ``model_results`` file in this directory.

**Training results:**

The training parameters used to train the models can be found in the ``train.py`` script 
of the speech commands folder. The models are trained to classify 8 keywords.

1) To train my convolutional model ``cnn`` and obtain number of parameters and test results,
 run the following in terminal:
```
python speech_commands/train.py --model_architecture='cnn'
```
2) To train my recurrent model ``rnn`` and obtain number of parameters and test results,
 run the following in terminal:
```
python speech_commands/train.py --model_architecture='rnn'
```
3) To train and obtain results of the baseline model, please do the following as it uses deprecated 
tensorflow libraries which does not run on my machine:

I) Open the Google Colab notebook at: 
[Tensorflow keyword recognition](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb?fbclid=IwAR02kZv7RoO36_OM7vqUQoTemdecxMBzAi8-wBBoI1MIxzgJcWPkrIcqAIg#scrollTo=ludfxbNIaegy) 

II) Change the training parameters in the notebook to match those of the ``train.py`` script in this project

III) Change the model architecture parameter in the notebook to ``conv``. Then train the model and obtain results.

- To train a quantized model (for eight bit deployment), change the ``quantize`` flag in the train.py script to 'True'

**Freezing trained model:**

In order to obtain a frozen representation of a model using the trained checkpoint from the previous step, do as follows:

1) To obtain the frozen graph of the ``cnn`` model, run the following in terminal:
```
python speech_commands/freeze.py --model_architecture='cnn' --start_checkpoint='/tmp/speech_commands_train/cnn.ckpt-30000' --output_file='/tmp/cnn'
```
2) 1) To obtain the frozen graph of the ``rnn`` model, run the following in terminal:
```
python speech_commands/freeze.py --model_architecture='rnn' --start_checkpoint='/tmp/speech_commands_train/rnn.ckpt-30000' --output_file='/tmp/rnn'
```
3) To obtain the frozen graph of the baseline model, run the corresponding cell of the Google Colab notebook provided previously.

**Test streaming accuracy:**

To test streaming performance of model on a synthetic audio file by using the frozen model from the previous step,
generate the audio file and its corresponding ground truth label file, run the following:
```
python speech_commands/generate_streaming_test_wav.py
```

1) To obtain steaming accuracy of ``cnn`` model on the generated audio file, run the following in terminal:
```
python speech_commands/test_streaming_accuracy.py --wav='/tmp/speech_commands_train/streaming_test.wav'  --ground-truth='/tmp/speech_commands_train/streaming_test_labels.txt' --labels='../labels.txt' --model='/tmp/cnn'
```
2) To obtain steaming accuracy of `rnn` model on the generated audio file, run the following in terminal:
```
python speech_commands/test_streaming_accuracy.py --wav='/tmp/speech_commands_train/streaming_test.wav'  --ground-truth='/tmp/speech_commands_train/streaming_test_labels.txt' --labels='../labels.txt' --model='/tmp/rnn'
```




