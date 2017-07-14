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

import numpy as np
import os

import data
import language_check


class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, token_strings, log_probs, state, attn_dists, p_gens, coverage):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      tokens: List of strings.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
    self.tokens = tokens
    self.token_strings = token_strings
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage
    self._scores = {}


  def extend(self, token, token_string, log_prob, state, attn_dist, p_gen, coverage):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      token_string: string.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(
      tokens=self.tokens + [token],
      token_strings=self.token_strings + [token_string],
      log_probs=self.log_probs + [log_prob],
      state=state,
      attn_dists=self.attn_dists + [attn_dist],
      p_gens=self.p_gens + [p_gen],
      coverage=coverage,
    )


  @property
  def latest_token(self):
    return self.tokens[-1]


  def _is_early_malformed(self, vocab_size, stop_token_id, comma_id):
    if _has_unknown_token(self.tokens, stop_token_id):
      return True
    if language_check.has_repeated_n_gram(self.tokens):
      return True
    if _has_repeated_entity(self.tokens, self.token_strings, vocab_size, comma_id):
      return True
    if language_check.sent_end_on_preposition(self.token_strings):
      return True
    return False


  def score(self, vocab_size, key_token_ids, article_id_to_word_id, is_complete):
    if is_complete in self._scores:
      return self._scores[is_complete]

    if self._is_early_malformed(vocab_size, key_token_ids['stop'], key_token_ids['comma']):
      return -(10. ** 6)
    if is_complete and language_check.has_poor_grammar(self.token_strings):
      return -(10. ** 6)

    token_scores = []
    has_seen_period = False

    for i, (token, log_prob) in enumerate(zip(self.tokens, self.log_probs)):
      log_prob -= float(token in key_token_ids['pronouns'])
      dec_token = article_id_to_word_id.get(token, token)
      if not has_seen_period:
        log_prob += 6. / (i + 4.) * float(dec_token in key_token_ids['people'])
        log_prob += 3. / (i + 4.) * float(dec_token in key_token_ids['other_entities'])
        has_seen_period = token == key_token_ids['period']

      token_scores.append(log_prob)

    self._scores[is_complete] = sum(token_scores) / len(token_scores) - self.cov_loss
    return self._scores[is_complete]


  @property
  def cov_loss(self):
    coverage = np.zeros_like(self.attn_dists[0]) # shape (batch_size, attn_length).
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).

    for a in self.attn_dists:
      covloss = np.minimum(a, coverage).sum()  # calculate the coverage loss for this step
      covlosses.append(covloss)
      coverage += a  # update the coverage vector

    return sum(covlosses) / len(covlosses)


def get_key_token_ids(vocab):
  return {
    'stop': vocab.word2id(data.STOP_DECODING, None),
    'comma': vocab.word2id(',', None),
    'period': vocab.word2id('.', None),
    'pronouns': {vocab.word2id(word, None) for word in ('he', 'she', 'him', 'her')},
    'people': {vocab.word2id(word, None) for word in data.PERSON_TOKENS + ('[ORG]',)},
    'other_entities': {
      vocab.word2id(word, None)
      for word in data.ENTITY_TOKENS + ('[PROPN]',)
      if word not in data.PERSON_TOKENS and word != '[ORG]'
    },
  }


def run_beam_search(
  sess, model, vocab, batch, beam_size, max_dec_steps, min_dec_steps, trace_path=''
):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  article_id_to_word_id = batch.article_id_to_word_ids[0]
  # Run the encoder to get the encoder hidden states and decoder initial state
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*enc_hidden_dim].
  enc_states, dec_in_state = model.run_encoder(sess, batch)

  # Initialize beam_size-many hyptheses
  hyps = [
    Hypothesis(
      tokens=[vocab.word2id(data.START_DECODING, None)],
      token_strings=[data.START_DECODING],
      log_probs=[0.],
      state=dec_in_state,
      attn_dists=[],
      p_gens=[],
      coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
    )
    for _ in xrange(beam_size)
  ]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)
  key_token_ids = get_key_token_ids(vocab)

  steps = 0
  while steps < max_dec_steps and len(results) < 4 * beam_size:
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    latest_tokens = [article_id_to_word_id.get(t, t) for t in latest_tokens]
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)

    # Run one step of the decoder to get the new info
    topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage = model.decode_onestep(
      sess=sess,
      batch=batch,
      latest_tokens=latest_tokens,
      enc_states=enc_states,
      dec_init_states=states,
      prev_coverage=prev_coverage,
    )

    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in xrange(num_orig_hyps):
      h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  # take the ith hypothesis and new decoder state info
      for j in xrange(2 * beam_size):  # for each of the top 2 * beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(
          token=topk_ids[i, j],
          token_string=data.outputid_to_word(topk_ids[i, j], vocab, batch.art_oovs[0]),
          log_prob=topk_log_probs[i, j],
          state=new_state,
          attn_dist=attn_dist,
          p_gen=p_gen,
          coverage=new_coverage_i,
        )
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(
      all_hyps, vocab.size, key_token_ids, article_id_to_word_id, complete_hyps=False
    ):
      # in order of most likely h
      if h.latest_token == vocab.word2id(data.STOP_DECODING, None): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= min_dec_steps:
          results.append(h)
      elif h.latest_token >= data.N_FREE_TOKENS:
        # hasn't reached stop token and generated non-unk token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == beam_size or len(results) == 4 * beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  if trace_path:
    for i, trace in enumerate(model._traces):
      with open(os.path.join(trace_path, 'timeline_%d.json' % i), 'w') as f:
        f.write(trace)

  # At this point, either we've got 4 * beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(
    results, vocab.size, key_token_ids, article_id_to_word_id, complete_hyps=True
  )

  # Return the hypothesis with highest average log prob
  best_hyp = hyps_sorted[0]
  return best_hyp, best_hyp.score(vocab.size, key_token_ids, article_id_to_word_id, is_complete=True)


def sort_hyps(hyps, vocab_size, key_token_ids, article_id_to_word_id, complete_hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(
    hyps,
    key=lambda h: h.score(vocab_size, key_token_ids, article_id_to_word_id, complete_hyps),
    reverse=True
  )


def _has_unknown_token(tokens, stop_token_id):
  if any(token < data.N_FREE_TOKENS for token in tokens[1:-1]):
    return True
  latest_token = tokens[-1]
  if latest_token < data.N_FREE_TOKENS and latest_token != stop_token_id:
    return True

  return False


def _has_repeated_entity(tokens, token_strings, vocab_size, comma_id):
  for i in range(len(tokens) - 2):
    if tokens[i] < vocab_size:
      continue
    if token_strings[i] == token_strings[i + 1]:
      return True
    if token_strings[i] == token_strings[i + 2] and tokens[i + 1] == comma_id:
      return True

  if tokens[-2] >= vocab_size and token_strings[-2] == token_strings[-1]:
    return True

  return False
