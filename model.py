"""
Builds and runs the tensorflow graph for the sequence-to-sequence model.
"""

import numpy as np
import os
import sys
import tensorflow as tf
import time
from collections import namedtuple
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import timeline

from attention_decoder import attention_decoder
from data import N_FREE_TOKENS, N_IMPORTANT_TOKENS, START_DECODING


Settings = namedtuple('Settings', (
    'embeddings_path',
    'log_root',
    'trace_path',
))


Hps = namedtuple('Hyperparameters', (
    'adagrad_init_acc',
    'adam_optimizer',
    'attn_only_entities',
    'batch_size',
    'copy_common_loss_wt',
    'copy_only_entities',
    'cov_loss_wt',
    'coverage',
    'dec_hidden_dim',
    'emb_dim',
    'enc_hidden_dim',
    'high_attn_loss_wt',
    'lr',
    'max_dec_steps',
    'max_enc_steps',
    'max_grad_norm',
    'mode',
    'output_vocab_size',
    'people_loss_wt',
    'rand_unif_init_mag',
    'restrictive_embeddings',
    'save_matmul',
    'tied_output',
    'trunc_norm_init_std',
    'two_layer_lstm',
))


class SummarizationModel(object):
    """
    A class to represent a sequence-to-sequence model for text summarization. Supports 
    pointer-generators and coverage, as well as modes for training, evaluating, and decoding.
    """

    def __init__(self, settings, hps, vocab):
        self._settings = settings
        self._hps = hps
        self._vocab = vocab
        self._traces = []


    def build_graph(self):
        """
        Add the placeholders, model, global step, train_op and summaries to the graph.
        """
        tf.logging.info('Building graph...')
        tf.summary.scalar('max_enc_steps', self._hps.max_enc_steps)
        tf.summary.scalar('max_dec_steps', self._hps.max_dec_steps)
        t0 = time.time()

        self._add_placeholders()
        self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()

        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)


    def _add_placeholders(self):
        """
        Add placeholders to the graph. These are entry points for any input data.
        """
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_batch_extend_vocab = tf.placeholder(
            tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab'
        )
        self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # decoder part
        self._dec_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch'
        )
        self._target_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch'
        )
        self._padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, hps.max_dec_steps], name='padding_mask'
        )
        self._padding_mask_people = tf.placeholder(
            tf.float32, [hps.batch_size, hps.max_dec_steps], name='padding_mask_people'
        )
        self._people_lens = tf.placeholder(tf.int32, [hps.batch_size], name='people_lens')
        self._people_ids = tf.placeholder(tf.int32, [hps.batch_size, None], name='people_ids')

        if hps.mode == "decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(
                tf.float32, [hps.batch_size, None], name='prev_coverage'
            )


    def _make_feed_dict(self, batch, just_enc=False, dec_batch=None):
        """
        Make a feed dictionary mapping parts of the batch to the appropriate placeholders.
    
        Args:
            batch: Batch object
            just_enc: Boolean. If True, only feed the parts needed for the encoder.
            dec_batch: tensor of shape [batch_size, max_dec_steps]. If provided, overrides
                batch.dec_batch.
        """
        if dec_batch is None:
            dec_batch = batch.dec_batch

        feed_dict = {
            self._enc_batch: batch.enc_batch,
            self._enc_lens: batch.enc_lens,
            self._enc_batch_extend_vocab: batch.enc_batch_extend_vocab,
            self._max_art_oovs: batch.max_art_oovs,
        }

        if not just_enc:
            feed_dict[self._dec_batch] = dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._padding_mask] = batch.padding_mask
            feed_dict[self._padding_mask_people] = batch.padding_mask_people
            feed_dict[self._people_lens] = batch.people_lens
            feed_dict[self._people_ids] = batch.people_ids

        return feed_dict


    def _add_seq2seq(self):
        """
        Add the whole sequence-to-sequence model to the graph.
        """
        hps = self._hps

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(
                -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123
            )
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            entity_tokens = tf.logical_and(
                tf.greater_equal(self._enc_batch, 3),
                tf.less(self._enc_batch, N_IMPORTANT_TOKENS),
            )
            self._entity_tokens = tf.to_float(entity_tokens)

            # Add embedding matrix (shared by the encoder and decoder inputs).
            with tf.variable_scope('embedding'):
                embedding = self._add_embeddings()

                if hps.mode == "train":
                    # add to tensorboard
                    self._add_emb_vis(embedding)

                # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)

                # list length max_dec_steps containing shape (batch_size, emb_size)
                emb_dec_inputs = [
                    tf.nn.embedding_lookup(embedding, x)
                    for x in tf.unstack(self._dec_batch, axis=1)
                ]

            # Add the encoder.
            with tf.variable_scope('encoder'):
                enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs

            # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce
            # the final encoder hidden state to the right size to be the initial decoder hidden
            # state.
            with tf.variable_scope('reduce_final_st'):
                self._dec_in_state = self._reduce_states(fw_st, bw_st)
            if hps.two_layer_lstm:
                with tf.variable_scope('reduce_final_st_top'):
                    top_dec_in_state = self._reduce_states(fw_st, bw_st)
                # tuple of states, one value per layer
                self._dec_in_state = self._dec_in_state, top_dec_in_state

            # Add the decoder.
            with tf.variable_scope('decoder'):
                (
                    decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens,
                    self.coverage
                ) = self._add_decoder(emb_dec_inputs)

            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('output_projection'):
                vocab_dists = self._add_projection(embedding, decoder_outputs)

            # Calc final distribution from copy distribution and vocabulary distribution.
            with tf.variable_scope('final_distribution'):
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
                # Take log of final distribution.
                log_dists = [tf.log(dist) for dist in final_dists]

            if hps.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.variable_scope('loss'):
                    self._add_loss(log_dists)

        if hps.mode == "decode":
            # We run decode beam search mode one decoder step at a time.
            # log_dists is a singleton list containing shape (batch_size, extended_vsize).
            assert len(log_dists) == 1
            log_dists = log_dists[0]
            # note batch_size = beam_size in decode mode
            self._topk_log_probs, self._topk_ids = tf.nn.top_k(log_dists, hps.batch_size*2)
        else:
            # Used to get output words to be fed back for training
            # shape [max_dec_steps, batch_size, 4].
            self._topk_log_probs, self._topk_ids = tf.nn.top_k(log_dists, 4)


    def _add_embeddings(self):
        """
        Add the embedding layer, depending upon whether we want to initialize them with pretrained
        embeddings and whether to restrict the embedding layer to a linear transform of the
        pretrained embeddings.
        
        Returns embedding variable of shape (vsize, emb_dim).
        """
        hps = self._hps
        vsize = self._vocab.size
        embeddings_path = self._settings.embeddings_path

        if embeddings_path:
            tf.logging.info('Using pretrained embeddings')
            embedding_value = np.load(embeddings_path)

            if hps.restrictive_embeddings:
                assert embedding_value.shape[0] == vsize
                embedding_transform = tf.get_variable(
                    'embedding_transform',
                    shape=[embedding_value.shape[1], hps.emb_dim],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init,
                )
                known_token_embeddings = tf.matmul(embedding_value, embedding_transform)
                unknown_token_embeddings = tf.get_variable(
                    'unknown_token_embeddings',
                    shape=[N_FREE_TOKENS, hps.emb_dim],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init,
                )
                unknown_token_embeddings_full = tf.pad(
                    unknown_token_embeddings, [[0, vsize - N_FREE_TOKENS], [0, 0]]
                )
                embedding = known_token_embeddings + unknown_token_embeddings_full

            else:
                assert embedding_value.shape == (vsize, hps.emb_dim)
                embedding = tf.Variable(
                    name='embedding',
                    initial_value=embedding_value,
                    dtype=tf.float32,
                )

        else:
            embedding = tf.get_variable(
                'embedding',
                shape=[vsize, hps.emb_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init,
            )

        return embedding


    def _add_emb_vis(self, embedding_var):
        """
        Do setup so that we can view word embedding visualization in Tensorboard, as described
        here: https://www.tensorflow.org/get_started/embedding_viz. Make the vocab metadata file,
        then make the projector config file pointing to it.
        """
        train_dir = os.path.join(self._settings.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path) # write metadata file
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)


    def _add_encoder(self, encoder_inputs, seq_len):
        """
        Add a single-layer bidirectional LSTM encoder to the graph.
    
        Args:
            encoder_inputs:
                A tensor of shape [batch_size, <=max_enc_steps, emb_size].
            seq_len:
                Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].
    
        Returns:
            encoder_outputs:
                A tensor of shape [batch_size, <=max_enc_steps, 2*enc_hidden_dim]. It's 
                2 * enc_hidden_dim because it's the concatenation of the forwards and backwards
                states.
            fw_state, bw_state:
                Each are LSTMStateTuples of shape
                ([batch_size, enc_hidden_dim], [batch_size, enc_hidden_dim]).
        """
        cell_fw = tf.contrib.rnn.LSTMCell(
            self._hps.enc_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True
        )
        cell_bw = tf.contrib.rnn.LSTMCell(
            self._hps.enc_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True
        )

        encoder_outputs, (fw_st, bw_st) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len,
            swap_memory=True
        )
        # concatenate the forwards and backwards states
        encoder_outputs = tf.concat(axis=2, values=encoder_outputs)

        if self._hps.two_layer_lstm:
            # Run one more bidirectional rnn that takes in the concatenated outputs of the
            # previous rnn as input.
            cell_fw_top = tf.contrib.rnn.LSTMCell(
                self._hps.enc_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True
            )
            cell_bw_top = tf.contrib.rnn.LSTMCell(
                self._hps.enc_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True
            )
            encoder_outputs, (fw_st, bw_st) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw_top, cell_bw_top, encoder_outputs, dtype=tf.float32,
                sequence_length=seq_len, swap_memory=True, scope='layer_two'
            )
            # concatenate the forwards and backwards states
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)

        return encoder_outputs, fw_st, bw_st


    def _reduce_states(self, fw_st, bw_st):
        """
        Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single
        initial state for the decoder. This is needed because the encoder is bidirectional but the
        decoder is not.
    
        Args:
            fw_st: LSTMStateTuple with enc_hidden_dim units.
            bw_st: LSTMStateTuple with enc_hidden_dim units.
    
        Returns:
            state: LSTMStateTuple with dec_hidden_dim units.
        """
        enc_hidden_dim = self._hps.enc_hidden_dim
        dec_hidden_dim = self._hps.dec_hidden_dim

        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable(
            'w_reduce_c', [2 * enc_hidden_dim, dec_hidden_dim], dtype=tf.float32,
            initializer=self.trunc_norm_init
        )
        w_reduce_h = tf.get_variable(
            'w_reduce_h', [2 * enc_hidden_dim, dec_hidden_dim], dtype=tf.float32,
            initializer=self.trunc_norm_init
        )
        bias_reduce_c = tf.get_variable(
            'bias_reduce_c', [dec_hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init
        )
        bias_reduce_h = tf.get_variable(
            'bias_reduce_h', [dec_hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init
        )

        # Apply linear layer
        # Concatenation of fw and bw cell
        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
        # Concatenation of fw and bw state
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
        # Get new cell from old cell
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
        # Get new state from old state
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)

        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)


    def _add_decoder(self, inputs):
        """
        Add attention decoder to the graph. In train or eval mode, you call this once to get output
        on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.
    
        Args:
            inputs: inputs to the decoder (word embeddings). A list of tensors shape
                (batch_size, emb_dim)
    
        Returns:
            outputs: List of tensors; the outputs of the decoder
            out_state: The final state of the decoder
            attn_dists: A list of tensors; the attention distributions
            p_gens: A list of scalar tensors; the generation probabilities
            coverage: A tensor, the current coverage vector
        """
        hps = self._hps
        if hps.two_layer_lstm:
            cells = [
                tf.contrib.rnn.LSTMCell(
                    hps.dec_hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init
                )
                for i in range(2)
            ]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.LSTMCell(
                hps.dec_hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init
            )

        # In decode mode, we run attention_decoder one step at a time and so need to pass in the
        # previous step's coverage vector each time
        prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None

        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(
            inputs,
            self._dec_in_state,
            self._enc_states,
            cell,
            initial_state_attention=(hps.mode == "decode"),
            use_coverage=hps.coverage,
            prev_coverage=prev_coverage,
            entity_tokens=self._entity_tokens if hps.attn_only_entities else None,
        )

        return outputs, out_state, attn_dists, p_gens, coverage


    def _add_projection(self, embedding, decoder_outputs):
        """
        Add the projection layer for the generated output distribution. Returns length
        max_dec_steps list of distributions of shape (batch_size, vsize).
        
        Args:
            embedding: variable of shape (vsize, emb_dim)
            decoder_outputs: list of decoder outputs of shape (batch_size, input_size)
        """
        hps = self._hps
        vsize = self._vocab.size

        if hps.save_matmul:
            assert hps.tied_output
            # Use precomputed projection matrix.
            w_full = tf.get_variable(
                'w_full', [hps.dec_hidden_dim, hps.output_vocab_size], dtype=tf.float32,
                initializer=self.trunc_norm_init
            )
        elif hps.tied_output:
            # Projection matrix is a matrix product of our projection variable and the
            # embeddings.
            w = tf.get_variable(
                'w', [hps.dec_hidden_dim, hps.emb_dim], dtype=tf.float32,
                initializer=self.trunc_norm_init
            )
            truncated_embedding = tf.slice(
                embedding, [0, 0], [hps.output_vocab_size, hps.emb_dim]
            )
            w_full = tf.matmul(w, truncated_embedding, transpose_b=True)
        else:
            w_full = tf.get_variable(
                'w', [hps.dec_hidden_dim, hps.output_vocab_size], dtype=tf.float32,
                initializer=self.trunc_norm_init
            )

        v = tf.get_variable(
            'v', [hps.output_vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init
        )
        # vocab_scores is the vocabulary distribution before applying softmax. Each entry
        # on the list corresponds to one decoder step
        vocab_scores = []
        for i, output in enumerate(decoder_outputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # apply the linear layer
            gen_output = tf.nn.xw_plus_b(output, w_full, v)
            if hps.output_vocab_size < vsize:
                gen_output = tf.pad(
                    gen_output, [[0, 0], [0, vsize - hps.output_vocab_size]]
                )
            vocab_scores.append(gen_output)

        # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize)
        # arrays. The words are in the order they appear in the vocabulary file.
        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
        return vocab_dists


    def _calc_final_dist(self, vocab_dists, attn_dists):
        """
        Calculate the final distribution, for the pointer-generator model
    
        Args:
            vocab_dists:
                The vocabulary distributions. List length max_dec_steps of (batch_size, vsize)
                arrays. The words are in the order they appear in the vocabulary file.
            attn_dists:
                The attention distributions. List length max_dec_steps of (batch_size, attn_len)
                arrays.
    
        Returns:
            final_dists:
                The final distributions for output words. List length max_dec_steps of 
                (batch_size, extended_vsize) tensors.
        """
        # Multiply vocab dists by p_gen and attention dists by (1 - p_gen)
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]

        if self._hps.copy_only_entities:
            attn_dists = [self._entity_tokens * dist for dist in attn_dists]
            attn_sums = [tf.reduce_sum(dist, axis=1, keep_dims=True) for dist in attn_dists]
            attn_dists = [dist / sum_ for dist, sum_ in zip(attn_dists, attn_sums)]

        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for
        # in-article OOV words

        # the maximum (over the batch) size of the extended vocabulary
        extended_vsize = self._vocab.size + self._max_art_oovs
        extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
        # list length max_dec_steps of shape (batch_size, extended_vsize)
        vocab_dists_extended = [
            tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists
        ]

        # Project the values in the attention distributions onto the appropriate entries in
        # the final distributions. This means that if a_i = 0.1 and the ith encoder word is w,
        # and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the
        # final distribution. This is done for each decoder timestep.

        # This is fiddly; we use tf.scatter_nd to do the projection.
        batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
        indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
        shape = [self._hps.batch_size, extended_vsize]
        # list length max_dec_steps (batch_size, extended_vsize)
        self.attn_dists_projected = [
            tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists
        ]

        # Add the vocab distributions and the copy distributions together to get the final
        # distributions.
        # final_dists is a list length max_dec_steps; each entry is a tensor shape
        # (batch_size, extended_vsize) giving the final distribution for that decoder timestep.
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this
        # is junk - ignore.
        final_dists = [
            vocab_dist + copy_dist
            for (vocab_dist, copy_dist) in zip(vocab_dists_extended, self.attn_dists_projected)
        ]

        # OOV part of vocab is max_art_oov long. Not all the sequences in a batch will have
        # max_art_oov tokens. That will cause some entries to be 0 in the distribution, which
        # will result in NaN when calulating log_dists.
        # Add a very small number to prevent that.
        def add_epsilon(dist, epsilon=sys.float_info.epsilon):
            epsilon_mask = tf.ones_like(dist) * epsilon
            return dist + epsilon_mask

        final_dists = [add_epsilon(dist) for dist in final_dists]

        return final_dists


    def _add_loss(self, log_dists):
        """
        Compute losses:
            log_prob per step
            (if enabled) people_log_prob for people steps
            (if enabled) coverage loss
            (if enabled) high attention loss
        
        Args:
            log_dists: List length max_dec_steps of (batch_size, extended_vsize)
        """
        hps = self._hps

        # Will be list length max_dec_steps containing shape (batch_size).
        loss_per_step = []
        # Will be list length max_dec_steps containing shape (batch_size).
        incorrect_people_loss_per_step = []
        batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)

        for dec_step, log_dist in enumerate(log_dists):
            # The indices of the target words. shape (batch_size)
            targets = self._target_batch[:, dec_step]
            # shape (batch_size, 2)
            indices = tf.stack((batch_nums, targets), axis=1)
            # shape (batch_size). loss on this step for each batch
            losses = tf.gather_nd(-log_dist, indices)
            loss_per_step.append(losses)

            if hps.people_loss_wt:
                incorrect_people_loss_per_batch = []
                for batch_num in range(hps.batch_size):
                    batch_people_len = self._people_lens[batch_num]
                    # Indices of all people for this batch. shape (batch_people_len)
                    batch_people_ids = self._people_ids[batch_num, :batch_people_len]
                    # shape (batch_people_len)
                    batch_indices = batch_num * tf.ones_like(
                        batch_people_ids, dtype=tf.int32
                    )
                    # shape (batch_people_len, 2)
                    people_loss_indices = tf.stack(
                        (batch_indices, batch_people_ids), axis=1
                    )
                    # shape (batch_people_len)
                    people_losses = tf.gather_nd(-log_dist, people_loss_indices)
                    # shape ()
                    people_losses = tf.reduce_mean(people_losses)
                    # convert nan to 0 if necessary.
                    people_losses = tf.where(
                        tf.is_nan(people_losses),
                        tf.zeros_like(people_losses),
                        people_losses
                    )
                    incorrect_people_loss_per_batch.append(people_losses)
                incorrect_people_loss_per_step.append(incorrect_people_loss_per_batch)

        # Apply padding_mask mask and get loss
        self._output_loss = _mask_and_avg(loss_per_step, self._padding_mask)
        tf.summary.scalar('loss', self._output_loss)

        self._people_loss = 0.
        self._coverage_loss = 0.
        self._high_attn_loss = 0.
        self._copy_common_loss = 0.

        if hps.people_loss_wt:
            # Calculate people losses
            with tf.variable_scope('people_loss'):
                correct_people_loss = _mask_and_avg(
                    loss_per_step, self._padding_mask_people, equal_wt_per_ex=False
                )
                other_people_loss = _mask_and_avg(
                    incorrect_people_loss_per_step, self._padding_mask_people,
                    equal_wt_per_ex=False
                )
                people_loss = .1 * correct_people_loss - .1 * other_people_loss
                tf.summary.scalar('people_loss', people_loss)
                self._people_loss = hps.people_loss_wt * people_loss

        if hps.coverage:
            # Calculate coverage loss from the attention distributions
            with tf.variable_scope('coverage_loss'):
                coverage_loss = _coverage_loss(self.attn_dists, self._padding_mask)
                tf.summary.scalar('coverage_loss', coverage_loss)
                self._coverage_loss = hps.cov_loss_wt * coverage_loss

        if hps.high_attn_loss_wt:
            # Calculate loss for high attention to non-entity words
            with tf.variable_scope('high_attn_loss'):
                high_attn_loss = _high_attn_loss(
                    self.attn_dists, self._padding_mask, self._entity_tokens
                )
                tf.summary.scalar('high_attn_loss', high_attn_loss)
                self._high_attn_loss = hps.high_attn_loss_wt * high_attn_loss

        if hps.copy_common_loss_wt:
            # Calculate loss for copying a common word (according to attention)
            with tf.variable_scope('copy_common_loss'):
                copy_common_loss = _copy_common_loss(
                    self.attn_dists_projected, log_dists, self._padding_mask, self._vocab.size,
                    self._hps.batch_size
                )
                tf.summary.scalar('copy_common_loss', copy_common_loss)
                self._copy_common_loss = hps.copy_common_loss_wt * copy_common_loss

        self._total_loss = (
            self._output_loss + self._people_loss + self._coverage_loss + self._high_attn_loss +
            self._copy_common_loss
        )


    def _add_train_op(self):
        """
        Sets self._train_op, the op to run for training.
        """
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        tvars = tf.trainable_variables()
        gradients = tf.gradients(
            self._total_loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
        )

        # Clip the gradients
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply optimizer
        if self._hps.adam_optimizer:
            optimizer = tf.train.AdamOptimizer()
        else:
            optimizer = tf.train.AdagradOptimizer(
                self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc
            )

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step'
        )


    def run_train_step(self, sess, batch, use_generated_inputs):
        """
        Runs one training iteration. Returns a dictionary containing train op, summaries, loss,
        global_step and (optionally) coverage loss.
        """
        feed_dict, to_return = self._get_train_step_feed_return(
            batch, get_outputs=use_generated_inputs
        )
        results = sess.run(to_return, feed_dict)
        if not use_generated_inputs:
            return results

        tf.logging.info('running training step with generated input...')
        t0 = time.time()

        # Run a second training step, where the inputs at each step can be either the real
        # labeled input or an input sampled from the output distribution of the model.
        output_ids = self._get_sampled_decoded_output(batch, results)
        feed_dict_generated, to_return_generated = self._get_train_step_feed_return(
            batch, get_outputs=False, output_ids=output_ids
        )
        results_generated = sess.run(to_return_generated, feed_dict_generated)

        tf.logging.info('seconds for training step with generated input: %.3f', time.time() - t0)
        tf.logging.info('generated-input loss: %f', results_generated['loss'])
        if self._hps.coverage:
            tf.logging.info("generated-input coverage_loss: %f", results_generated['coverage_loss'])

        return results


    def _get_sampled_decoded_output(self, batch, results):
        """
        Only used during corrective training. Returns sampled output from the model
        (with probability .2) to be used as input for a second iteration of training on the same
        batch.
        """
        probs = np.exp(results['top_k_log_probs'])
        ids = results['top_k_ids']
        assert probs.shape == (self._hps.max_dec_steps, self._hps.batch_size, 4)
        output_ids = np.empty((self._hps.batch_size, self._hps.max_dec_steps), dtype=np.int32)

        for i in range(self._hps.batch_size):
            output_ids[i, 0] = self._vocab.word2id(START_DECODING, None)
            for t in range(self._hps.max_dec_steps - 1):
                if np.random.random() < .2:
                    # sample from output distribution, and assign to output
                    prob_dist = probs[t, i] / probs[t, i].sum()
                    output_id = np.asscalar(np.random.choice(ids[t, i], size=1, p=prob_dist))
                    output_ids[i, t + 1] = batch.article_id_to_word_ids[i].get(output_id, output_id)
                else:
                    # use labeled output
                    output_ids[i, t + 1] = batch.dec_batch[i, t + 1]

        return output_ids


    def _get_train_step_feed_return(self, batch, get_outputs, output_ids=None):
        """
        Specify input and output of a training step.
        """
        assert not (get_outputs and output_ids)

        feed_dict = self._make_feed_dict(batch, dec_batch=output_ids)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._output_loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
            to_return['attn_dists'] = self.attn_dists
        if get_outputs:
            to_return['top_k_ids'] = self._topk_ids
            to_return['top_k_log_probs'] = self._topk_log_probs

        return feed_dict, to_return


    def run_eval_step(self, sess, batch):
        """
        Runs one evaluation iteration. Returns a dictionary containing summaries, loss,
        global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._output_loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)


    def run_encoder(self, sess, batch):
        """
        For beam search decoding. Run the encoder on the batch and return the encoder states and
        decoder initial state.
    
        Args:
            sess: Tensorflow session.
            batch: Batch object that is the same example repeated across the batch (for beam search)
    
        Returns:
            enc_states:
                The encoder states. A tensor of shape
                [batch_size, <= max_enc_steps, 2 * enc_hidden_dim].
            dec_in_state:
                A LSTMStateTuple of shape ([1, dec_hidden_dim],[1, dec_hidden_dim]). If two layers,
                then a tuple of such LSTMStateTuples.
                
        """
        # Feed the batch into the placeholders
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        # Run the encoder
        enc_states, dec_in_state, global_step = sess.run(
            [self._enc_states, self._dec_in_state, self.global_step], feed_dict
        )

        # dec_in_state is LSTMStateTuple shape
        # ([batch_size, dec_hidden_dim], [batch_size, dec_hidden_dim]).
        # Given that the batch is a single example repeated, dec_in_state is identical across the
        # batch so we just take the top row.
        if self._hps.two_layer_lstm:
            dec_in_state = tuple(
                tf.contrib.rnn.LSTMStateTuple(dec_in_state[l].c[0], dec_in_state[l].h[0])
                for l in range(2)
            )
        else:
            dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])

        return enc_states, dec_in_state


    def decode_onestep(
        self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage
    ):
        """
        For beam search decoding. Run the decoder for one step.
    
        Args:
            sess:
                Tensorflow session.
            batch:
                Batch object containing single example repeated across the batch
            latest_tokens:
                Tokens to be fed as input into the decoder for this timestep
            enc_states:
                The encoder states.
            dec_init_states:
                List of beam_size LSTMStateTuples; the decoder states from the previous timestep.
                If two layers, each state is instead a tuple of LSTMStateTuples.
            prev_coverage:
                List of np arrays. The coverage vectors from the previous timestep. List of None
                if not using coverage.
    
        Returns:
            ids:
                top 2k ids. shape [beam_size, 2*beam_size]
            probs:
                top 2k log probabilities. shape [beam_size, 2*beam_size]
            new_states:
                new states of the decoder. a list length beam_size containing LSTMStateTuples
                each of shape ([dec_hidden_dim,],[dec_hidden_dim,]). If two layers, each state
                is instead a tuple of LSTMStateTuples.
            attn_dists:
                List length beam_size containing lists length attn_length.
            p_gens:
                Generation probabilities for this step. A list length beam_size. List of None
                if in baseline mode.
            new_coverage:
                Coverage vectors for this step. A list of arrays. List of None if coverage is
                not turned on.
        """

        beam_size = len(dec_init_states)

        if not self._hps.two_layer_lstm:
            dec_init_states = [tuple([state]) for state in dec_init_states]

        # Turn dec_init_states (a list of tuple of LSTMStateTuples) into a single LSTMStateTuple
        # (or tuple of LSTMStateTuples) for the batch.
        n_layers = 2 if self._hps.two_layer_lstm else 1
        new_dec_in_states = []
        for l in range(n_layers):
            cells = [np.expand_dims(state[l].c, axis=0) for state in dec_init_states]
            hiddens = [np.expand_dims(state[l].h, axis=0) for state in dec_init_states]
            new_c = np.concatenate(cells, axis=0)  # shape [batch_size, dec_hidden_dim]
            new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size, dec_hidden_dim]
            new_dec_in_states.append(tf.contrib.rnn.LSTMStateTuple(new_c, new_h))

        new_dec_in_state = tuple(new_dec_in_states) if self._hps.two_layer_lstm else new_dec_in_states[0]

        feed = {
            self._enc_batch: batch.enc_batch,
            self._enc_states: enc_states,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self._enc_batch_extend_vocab: batch.enc_batch_extend_vocab,
            self._max_art_oovs: batch.max_art_oovs,
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists,
            "p_gens": self.p_gens,
        }

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        # Run the decoder step
        if self._settings.trace_path:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            results = sess.run(
                to_return, feed_dict=feed, options=options, run_metadata=run_metadata
            )

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            self._traces.append(chrome_trace)
        else:
            results = sess.run(to_return, feed_dict=feed)

        # Convert results['states'] (a single lstmstatetuple) into a list of lstmstatetuple,
        # one for each hypothesis.
        if self._hps.two_layer_lstm:
            new_states = [
                tuple(
                    tf.contrib.rnn.LSTMStateTuple(
                        results['states'][l].c[i, :], results['states'][l].h[i, :]
                    )
                    for l in range(2)
                )
                for i in xrange(beam_size)
            ]
        else:
            new_states = [
                tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :])
                for i in xrange(beam_size)
            ]

        # Convert singleton list containing a tensor to a list of k arrays.
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        # Convert singleton list containing a tensor to a list of k arrays.
        assert len(results['p_gens']) == 1
        p_gens = results['p_gens'][0].tolist()
        assert all(len(gen_list) == 1 for gen_list in p_gens)
        p_gens = [gen_list[0] for gen_list in p_gens]

        # Convert the coverage tensor to a list length k containing the coverage vector for each
        # hypothesis.
        if self._hps.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in xrange(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask, equal_wt_per_ex=True):
    """
    Applies mask to values then returns overall average (a scalar).
  
    args:
        values: a list length max_dec_steps containing arrays shape (batch_size).
        padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
  
    returns:
        a scalar
    """
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    if equal_wt_per_ex:
        # shape batch_size. float32
        dec_lens = tf.reduce_sum(padding_mask, axis=1)
        # shape (batch_size); normalized value for each batch member
        values_per_ex = sum(values_per_step) / tf.maximum(1., dec_lens)
        return tf.reduce_mean(values_per_ex)
    else:
        total = tf.reduce_sum(sum(values_per_step))
        total_weight = tf.reduce_sum(padding_mask)
        return total / total_weight


def _coverage_loss(attn_dists, padding_mask):
    """
    Calculates the coverage loss from the attention distributions.
  
    args:
        attn_dists: the attention distributions for each decoder timestep. A list of length
            max_dec_steps containing shape (batch_size, attn_length)
        padding_mask: shape (batch_size, max_dec_steps).
  
    returns:
        coverage_loss: scalar
    """
    # shape (batch_size, attn_length). initial coverage is zero.
    coverage = tf.zeros_like(attn_dists[0])
    # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape
    # (batch_size).
    covlosses = []

    for a in attn_dists:
        # calculate the coverage loss for this step
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        covlosses.append(covloss)
        # update the coverage vector
        coverage += a

    return _mask_and_avg(covlosses, padding_mask)


def _high_attn_loss(attn_dists, padding_mask, entity_tokens):
    """
    Calculates a high attention loss from the attention distributions.
  
    args:
        attn_dists:
            the attention distributions for each decoder timestep. A list of length max_dec_steps
            containing shape (batch_size, attn_length)
        padding_mask:
            shape (batch_size, max_dec_steps).
        entity_tokens:
            indicator for which tokens are entities (1 is entity, 0 is not). Tensor of shape 
            (batch_size, attn_length)
  
    returns:
        high_attn_loss: scalar
    """
    non_entity_attns = [(1. - entity_tokens) * attn_dist for attn_dist in attn_dists]
    max_non_entity_attns = [tf.reduce_max(attn_dist, axis=1) for attn_dist in non_entity_attns]
    non_entity_losses = [max_attn ** 3 for max_attn in max_non_entity_attns]
    return _mask_and_avg(non_entity_losses, padding_mask)


def _copy_common_loss(attn_dists_projected, log_dists, padding_mask, vocab_size, batch_size):
    """
    Calculates a loss for copying common words according to the attention distributions.
  
    args:
        attn_dists_projected:
            List of length max_dec_steps containing shape (batch_size, extended_vsize). The
            attention distributions for each decoder timestep. 
        log_dists:
            List of length max_dec_steps containing shape (batch_size, extended_vsize). The log
            probabilities of generating each word.
        padding_mask:
            shape (batch_size, max_dec_steps).
        vocab_size:
            int.
        batch_size:
            int.
  
    returns:
        copy_common_loss: scalar
    """
    common_log_prob_per_step = []

    for attn_dist, log_dist in zip(attn_dists_projected, log_dists):
        # (batch_size, extended_vsize)
        attn_times_log_dist = tf.multiply(attn_dist, log_dist)
        # (batch_size, vocab_size)
        common_word_dist = tf.slice(attn_times_log_dist, [0, 0], [batch_size, vocab_size])
        # (batch_size)
        common_copy_log_prob = tf.reduce_sum(common_word_dist, axis=1)
        common_log_prob_per_step.append(common_copy_log_prob)

    return _mask_and_avg(common_log_prob_per_step, padding_mask)



