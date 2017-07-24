"""
This file contains code to process data into batches.
"""

import Queue
import numpy as np
import tensorflow as tf
import time
from google.protobuf import text_format
from random import shuffle
from threading import Thread

import data


class Example(object):
    """
    Class representing a train / val / test example for text summarization.
    """

    def __init__(self, article, abstract, vocab, hps):
        """
        Initializes the Example, performing tokenization and truncation to produce the encoder,
        decoder and target sequences, which are stored in self.
    
        Args:
            article: source text; a string. each token is separated by a single space.
            abstract: reference summary; a string. each token is separated by a single space.
            vocab: Vocabulary object
            hps: hyperparameters
        """
        self.hps = hps

        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING, None)
        stop_decoding = vocab.word2id(data.STOP_DECODING, None)

        # Process the article
        article_words = [data.parse_word(word) for word in article.split()]
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]

        # store the length after truncation but before padding
        self.enc_len = len(article_words)
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w, word_type) for w, word_type in article_words]

        # Process the abstract
        abstract_words = [data.parse_word(word) for word in abstract.split()]
        # list of word ids; OOVs are represented by the id for UNK token
        abs_ids = [vocab.word2id(w, word_type) for w, word_type in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, target_orig = self.get_dec_inp_targ_seqs(
            abs_ids, hps.max_dec_steps, start_decoding, stop_decoding
        )
        self.dec_len = len(self.dec_input)
        # Store a version of the enc_input where in-article OOVs are represented by their
        # temporary OOV id; also store the in-article OOVs words themselves
        self.enc_input_extend_vocab, self.article_oovs, self.article_id_to_word_id = (
            data.article2ids(article_words, vocab, hps.copy_only_entities)
        )

        # Get a version of the reference summary where in-article OOVs are represented by their
        # temporary article OOV id
        if hps.copy_only_entities:
            copyable_words = set(self.article_oovs)
        else:
            copyable_words = set([w for w, word_type in article_words])
        abs_ids_extend_vocab = data.abstract2ids(
            abstract_words, vocab, self.article_oovs, copyable_words,
            hps.output_vocab_size or vocab.size
        )
        # Set decoder target sequence that uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(
            abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding
        )

        # Compute a mask for which tokens are people.
        people_tokens = {vocab.word2id('', token) for token in data.PERSON_TOKENS}
        self.target_people = [float(token in people_tokens) for token in target_orig]

        # Get list of people ids
        self.people_ids = []
        for article_id, word_id in self.article_id_to_word_id.iteritems():
            if word_id in people_tokens:
                self.people_ids.append(article_id)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """
        Given the reference summary as a sequence of tokens, return the input sequence for the
        decoder, and the target sequence which we will use to calculate loss. The sequence will
        be truncated if it is longer than max_len. The input sequence must start with the start_id
        and the target sequence must end with the stop_id (but not if it's been truncated).
    
        Args:
            sequence: List of ids (integers)
            max_len: integer
            start_id: integer
            stop_id: integer
    
        Returns:
            inp: sequence length <=max_len starting with start_id
            target: sequence same length as input, ending with stop_id only if there was no
                truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]

        if len(inp) > max_len:
            # truncate
            inp = inp[:max_len]
            # no end_token
            target = target[:max_len]
        else:
            # end token
            target.append(stop_id)

        assert len(inp) == len(target)
        return inp, target


    def pad_decoder_inp_targ(self, max_len, pad_id):
        """
        Pad decoder input, target, and people sequences with pad_id up to max_len.
        """
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)
        while len(self.target_people) < max_len:
            self.target_people.append(0.)


    def pad_encoder_input(self, max_len, pad_id):
        """
        Pad the encoder input sequence with pad_id up to max_len.
        """
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)

        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    """
    Class representing a minibatch of train / val / test examples for text summarization.
    """

    def __init__(self, example_list, hps, vocab):
        """
        Turns the example_list into a Batch object.
    
        Args:
            example_list: List of Example objects
            hps: hyperparameters
            vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(data.PAD_TOKEN, None)
        self.stop_id = vocab.word2id(data.STOP_DECODING, None)
        self.init_encoder_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list)


    def init_encoder_seq(self, example_list, hps):
        """
        Initializes the following:
        
            self.enc_batch:
                numpy array of shape (batch_size, <=max_enc_steps) containing integer ids
                (all OOVs represented by UNK id), padded to length of longest sequence in the batch
            self.enc_lens:
                numpy array of shape (batch_size) containing integers. The (truncated) length of
                each encoder input sequence (pre-padding).
            self.max_art_oovs:
                maximum number of in-article OOVs in the batch
            self.art_oovs:
                list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
                Same as self.enc_batch, but in-article OOVs are represented by their temporary
                article OOV number.
            self.article_id_to_word_ids:
                list of maps from article OOV id to vocab word id.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because
        # we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len

        # Determine the max number of in-article OOVs in this batch
        self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        self.art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

        self.article_id_to_word_ids = [example.article_id_to_word_id for example in example_list]


    def init_decoder_seq(self, example_list, hps):
        """
        Initializes the following:
        
            self.dec_batch:
                numpy array of shape (batch_size, max_dec_steps), containing integer ids as input
                for the decoder, padded to max_dec_steps length.
            self.target_batch:
                numpy array of shape (batch_size, max_dec_steps), containing integer ids for the
                target sequence, padded to max_dec_steps length.
            self.padding_mask:
                numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s
                correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
            self.padding_mask_people:
                numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s
                correspond to people tokens.
            self.people_lens:
                list of integers, specifying how many people ids are in each example.
            self.people_ids:
                numpy array of shape (batch_size, max(people_lens), containing the ids of people
                tokens for each batch.
        """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch
        # (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding.
        # However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in
        # which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        self.padding_mask_people = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.people_lens = [len(ex.people_ids) for ex in example_list]
        self.people_ids = np.zeros((hps.batch_size, max(self.people_lens)), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.padding_mask_people[i, :] = ex.target_people[:]
            self.padding_mask[i] = (ex.target >= data.N_FREE_TOKENS) | (ex.target == self.stop_id)
            self.people_ids[i, :len(ex.people_ids)] = ex.people_ids[:]


    def store_orig_strings(self, example_list):
        """
        Store the original article and abstract strings in the Batch object
        """
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]


class Batcher(object):
    """
    A class to generate minibatches of data. Buckets examples together based on length of the
    encoder sequence.
    """

    # max number of batches the batch_queue can hold
    BATCH_QUEUE_MAX = 100

    def __init__(self, data_path, vocab, hps, single_pass):
        """
        Initialize the batcher. Start threads that process the data into batches.
    
        Args:
            data_path: tf.Example filepattern.
            vocab: Vocabulary object
            hps: hyperparameters
            single_pass: If True, run through the dataset exactly once (useful for when you want
                to run evaluation on the dev or test set). Otherwise generate random batches
                indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be
        # batched
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass or 'sample' in data_path:
            # just one thread, so we read through the dataset just once
            self._num_example_q_threads = 1
            # just one thread to batch examples
            self._num_batch_q_threads = 1
            # only load one batch's worth of examples before bucketing; this essentially means no
            # bucketing
            self._bucketing_cache_size = 1
            # this will tell us when we're finished reading the dataset
            self._finished_reading = False
        else:
            # num threads to fill example queue
            self._num_example_q_threads = 16
            # num threads to fill batch queue
            self._num_batch_q_threads = 4
            # how many batches-worth of examples to load into cache before bucketing
            self._bucketing_cache_size = 100

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in xrange(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in xrange(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:
            # We don't want a watcher in single_pass mode because the threads shouldn't run forever.
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()


    def next_batch(self):
        """
        Return a Batch from the batch queue.
    
        If mode='decode' then each batch contains a single example repeated beam_size-many times;
        this is necessary for beam search.
    
        Returns:
            batch: a Batch object, or None if we're in single_pass mode and we've exhausted the
            dataset.
        """
        if self._batch_queue.qsize() == 0:
            # If the batch queue is empty, print a warning
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, '
                'Input queue size: %i',
                self._batch_queue.qsize(),
                self._example_queue.qsize()
            )
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()
        return batch


    def fill_example_queue(self):
        """
        Reads data from file and processes into Examples which are then placed into the example
        queue.
        """
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                # read the next example from file. article and abstract are both strings.
                (article, abstract) = input_gen.next()
            except StopIteration:
                # if there are no more examples:
                tf.logging.info(
                    "The example generator for this example queue filling thread has exhausted "
                    "data."
                )
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread "
                        "is stopping."
                    )
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        "single_pass mode is off but the example generator is out of data; error."
                    )

            example = Example(article, abstract, self._vocab, self._hps)
            self._example_queue.put(example)


    def fill_batch_queue(self):
        """
        Takes Examples out of example queue, sorts them by encoder sequence length, processes into
        Batches and places them in the batch queue.
    
        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            if self._hps.mode != 'decode':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for k in xrange(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                # sort by length of encoder sequence
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)

                # Group the sorted Examples into batches, optionally shuffle the batches, and
                # place in the batch queue.
                batches = []
                for i in xrange(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:
                    # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:
                # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in xrange(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))


    def watch_threads(self):
        """
        Watch example queue and batch queue threads and restart if dead.
        """
        while True:
            time.sleep(60)

            for idx,t in enumerate(self._example_q_threads):
                if not t.is_alive():
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

            for idx,t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()


    def text_generator(self, example_generator):
        """
        Generates article and abstract text from tf.Example.
    
        Args:
            example_generator: a generator of tf.Examples from file. See data.example_generator.
        """
        while True:
            # e is a tf.Example
            e = example_generator.next()
            try:
                # the article text was saved under the key 'article' in the data files
                article_text = e.features.feature['article'].bytes_list.value[0]
                # the abstract text was saved under the key 'abstract' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[0]
            except ValueError:
                tf.logging.error(
                    'Failed to get article or abstract from example: %s',
                    text_format.MessageToString(e)
                )
                continue
            if len(article_text)==0:
                # See https://github.com/abisee/pointer-generator/issues/1
                tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield (article_text, abstract_text)
