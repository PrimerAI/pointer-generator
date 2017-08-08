This repository contains code for experimenting with seq-to-seq summarization models. It is inititally based off of https://github.com/abisee/pointer-generator (corresponding paper is at https://arxiv.org/abs/1704.04368). The code uses Tensorflow 1.1.

# Code organization

Here are the key files in the project:

## Model definition

model.py - defines most of the model graph.
attention_decoder.py - defines the decoder with attention.
data.py - defines the vocab set and how to handle temporary article tokens for copying.
batcher.py - processes the data set into batches of samples.

## Scripts
run_summarization.py - main script for training and evaluating. Also converts model to coverage.
make_datafiles.py - preprocesses the training data into a format useful for training.
scripts.py - assortment of scripts for compiling initial word vectors for a vocabulary and generating sample outputs.

## Generating summaries
decoder.py - contains top level method `generate_summary` for generating outputs.
model_parameters/ - contains one checkpoint of model parameters (as of 8/7/17).
beam_search.py - the top level method `generate_summary`uses the code here to search for the best summary output.
io_processing.py - the top level method `generate_summary` uses the code here to process the input and output.

# Running the code

## Dataset

Get the CNN / Dailymail data set from http://cs.nyu.edu/~kcho/DMQA/. (ask about new cables if interested). Training is about 10 times faster on GPU - to set up an AWS instance with GPU, see https://primer.atlassian.net/wiki/spaces/DEVOPS/pages/10686621/CloudFormation+Stacks+at+Primer#CloudFormationStacksatPrimer-CreateaDeepLearningSimpleInstanceStack.

Run `python make_datafiles.py <data_dir> <output_dir> <n_workers>`, where data_dir contains the directories with the CNN / Dailymail / new cables data, containing one file per document. The script will use Spacy and SpacyPeopleResolver to tokenize and label the articles into output subdirectories for each data set, also one file per document. Finally, we care about the `finished_files` directory which has a vocab list, train / eval / test datasets, and those data sets chunked into files with 1000 examples each. (See `compute_reduced_embeddings_original_vocab()` in `scripts.py` for how to generate pretrained embeddings for the generated vocab).

## Training

To train a model, run

```
python run_summarization.py --mode={train, eval, decode} --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

There are a ton of other configurations / variants / features - see run_summarization.py for the full list of FLAGS. The default parameters are pretty good (the model saved in `model_parameters` was trained with the default parameters except with `adam_optimizer` enabled and a `embeddings_path` given. 

You can view training progress using `tensorboard --logdir=/path/to/log/directory`. Training loss on the CNN / Dailymail data set typically converges at around 2.75 and on the combined CNN / Dailymail / new cables data set converges at around 3.0. It's also a good idea to run the `run_summarization.py` script with `mode=eval`, to see the loss on an evaluation data set.

Training can take between half a day to three days (or more!) depending on the configuration. For faster training, it's recommended to begin training with smaller values for `max_enc_steps` and `max_dec_steps` (say 150 and 75) before gradually increasing to the final values (as training loss starts leveling off).

## Generating
Once trained, point the path in `decoder.py` to the subdirectory with the saved weights (the `train` directory in the log directory specified during training). Calling `generate_summary` returns the summary for a given document.

# Experiments

Below is a list of ideas I attempted (again as of 8/7/17):

- Use pretrained word embeddings
- Create different input tokens for entities
- Create different input tokens for each part-of-speech for out-of-vocab words
- Create different input tokens for each person as decided by the people resolver
- Force output projection matrix to be a tied to the word embeddings
- Force embedding matrix to be a tied to the pretrained word embeddings
- Schedule sampling feeding in sampled generated output as input during training
- Use attention distribution as part of metric during beam search
- Prevent n-gram repetition & reduce pronouns during beam search
- Use adam optimizer
- Increase decoder LSTM hidden state size
- Increase weight on entity tokens during training (and add loss for incorrect entities)
- Reduce overall vocab size (since we have POS replacements)
- Reduce output vocab size (since we don’t use full vocab during decoding)
- Prevent copying non-entity words from the input
- Utilize cables for training data
- Ignore UNK tokens for training loss
- Two layer encoder
- Penalty for high attention on non-entity words

See flags in `run_summarization.py` for how to enable these, as well as default parameters to see the (approximately best settings I found).

# Results

Results for the current checkpoint saved in the `results/` directory.
