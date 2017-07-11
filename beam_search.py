# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding"""

import os
import tensorflow as tf
import numpy as np
import data

FLAGS = tf.app.flags.FLAGS

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage

  def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  def _has_unknown_token(self, stop_token_id):
    if any(token < data.N_FREE_TOKENS for token in self.tokens[1:-1]):
      return True
    if self.latest_token < data.N_FREE_TOKENS and self.latest_token != stop_token_id:
      return True

    return False

  def avg_log_prob(self, stop_token_id):
    if self._has_unknown_token(stop_token_id):
      return -10 ** 6
    return sum(self.log_probs) / len(self.tokens)

    """
    # Compute average log_prob per step. Weigh the generative and copy parts equally so that we
    # don't bias towards sequences of only copying (which have higher log_probs generally).
    gen_sum = sum(self.p_gens)
    gen_score = sum(
      p_gen / gen_sum * log_prob for p_gen, log_prob in zip(self.p_gens, self.log_probs)
    )
    copy_sum = sum(1. - p_gen for p_gen in self.p_gens)
    copy_score = sum(
      (1. - p_gen) / copy_sum * log_prob for p_gen, log_prob in zip(self.p_gens, self.log_probs)
    )
    return .5 * gen_score + .5 * copy_score
    """

  def score(self, vocab_size, key_token_ids):
    if self._has_unknown_token(key_token_ids['stop']):
      return -10 ** 6

    avg_log_prob = 1. / len(self.tokens) * sum(
      lp - .75 * int(token in key_token_ids['pronouns'])
      for lp, token in zip(self.log_probs, self.tokens)
    )
    repeated_entity_loss = self.repeated_entity_loss(vocab_size, key_token_ids['comma'])
    return avg_log_prob - self.repeated_n_gram_loss - self.cov_loss - repeated_entity_loss

  def final_score(self, vocab_size, key_token_ids):
    grammatical_loss = self.grammatical_loss(
      key_token_ids['left_parens'], key_token_ids['right_parens'], key_token_ids['quote']
    )
    return self.score(vocab_size, key_token_ids) - grammatical_loss

  @property
  def cov_loss(self):
    coverage = np.zeros_like(self.attn_dists[0]) # shape (batch_size, attn_length).
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).

    for a in self.attn_dists:
      covloss = np.minimum(a, coverage).sum()  # calculate the coverage loss for this step
      covlosses.append(covloss)
      coverage += a  # update the coverage vector

    return sum(covlosses) / len(covlosses)

  @property
  def repeated_n_gram_loss(self, disallowed_n=3):
    seen_n_grams = set()

    for i in range(len(self.tokens) - disallowed_n + 1):
      n_gram = tuple(self.tokens[i: i + disallowed_n])
      if n_gram in seen_n_grams:
        return 10. ** 6
      seen_n_grams.add(n_gram)

    return 0.

  def repeated_entity_loss(self, vocab_size, comma_id):
    for i in range(len(self.tokens) - 2):
      if self.tokens[i] < vocab_size:
        continue
      if self.tokens[i] == self.tokens[i + 1]:
        return 10. ** 6
      if self.tokens[i] == self.tokens[i + 2] and self.tokens[i + 1] == comma_id:
        return 10. ** 6

    if self.tokens[-2] >= vocab_size and self.tokens[-2] == self.tokens[-1]:
      return 10. ** 6

    return 0.

  def grammatical_loss(self, left_parens_id, right_parens_id, quote_id):
    has_open_left_parens = False
    quote_count = 0

    for token in self.tokens:
      if token == left_parens_id:
        if has_open_left_parens:
          return 10. ** 6
        else:
          has_open_left_parens = True
      elif token == right_parens_id:
        if has_open_left_parens:
          has_open_left_parens = False
        else:
          return 10. ** 6
      elif token == quote_id:
        quote_count += 1

    return quote_count % 2 == 0


def get_key_token_ids(vocab):
  return {
    'stop': vocab.word2id(data.STOP_DECODING, None),
    'comma': vocab.word2id(',', None),
    'left_parens': vocab.word2id('(', None),
    'right_parens': vocab.word2id(')', None),
    'quote': vocab.word2id('"', None),
    'pronouns': {
      vocab.word2id(word, None) for word in ('he', 'she', 'him', 'her', 'i', 'we')
    }
  }


def run_beam_search(sess, model, vocab, batch):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  import time
  t0 = time.time()
  enc_states, dec_in_state = model.run_encoder(sess, batch)
  print "Encoding time:", time.time() - t0
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*enc_hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING, None)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
                     ) for _ in xrange(FLAGS.beam_size)]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)
  key_token_ids = get_key_token_ids(vocab)

  step_times = []
  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < 4 * FLAGS.beam_size:
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    latest_tokens = [batch.article_id_to_word_ids[0].get(t, t) for t in latest_tokens]
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)

    t0 = time.time()
    # Run one step of the decoder to get the new info
    topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, sess_time = model.decode_onestep(
      sess=sess,
      batch=batch,
      latest_tokens=latest_tokens,
      enc_states=enc_states,
      dec_init_states=states,
      prev_coverage=prev_coverage,
    )
    step_times.append((time.time() - t0, sess_time))

    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in xrange(num_orig_hyps):
      h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  # take the ith hypothesis and new decoder state info
      for j in xrange(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i, j],
                           log_prob=topk_log_probs[i, j],
                           state=new_state,
                           attn_dist=attn_dist,
                           p_gen=p_gen,
                           coverage=new_coverage_i)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps, vocab.size, key_token_ids, complete_hyps=False): # in order of most likely h
      if h.latest_token == vocab.word2id(data.STOP_DECODING, None): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      elif h.latest_token >= data.N_FREE_TOKENS:
        # hasn't reached stop token and generated non-unk token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == FLAGS.beam_size or len(results) == 4 * FLAGS.beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  print "Avg step time:", sum(st[0] for st in step_times) / steps
  print "Avg sess time:", sum(st[1] for st in step_times) / steps

  if FLAGS.trace_path:
    for i, trace in enumerate(model._traces):
      with open(os.path.join(FLAGS.trace_path, 'timeline_%d.json' % i), 'w') as f:
        f.write(trace)

  # At this point, either we've got 4 * beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results, vocab.size, key_token_ids, complete_hyps=True)

  # Return the hypothesis with highest average log prob
  return hyps_sorted[0]

def sort_hyps(hyps, vocab_size, key_token_ids, complete_hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  if complete_hyps:
    score_func = lambda h: h.final_score(vocab_size, key_token_ids)
  else:
    score_func = lambda h: h.score(vocab_size, key_token_ids)

  return sorted(hyps, key=score_func, reverse=True)
