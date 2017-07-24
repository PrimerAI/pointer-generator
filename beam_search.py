"""
This file contains code to run beam search decoding.
"""

import numpy as np
import os

import data
import language_check


class Hypothesis(object):
    """
    Class to represent a hypothesis during beam search. Holds all the information needed for the
    hypothesis.
    """

    def __init__(self, tokens, token_strings, log_probs, state, attn_dists, p_gens, coverage):
        """
        Hypothesis constructor.
    
        Args:
            tokens:
                List of integers. The ids of the tokens that form the summary so far.
            tokens:
                List of strings. The strings of the tokens so far.
            log_probs:
                List, same length as tokens, of floats, giving the log probabilities of the tokens
                so far.
            state:
                Current state of the decoder, a LSTMStateTuple.
            attn_dists:
                List, same length as tokens, of numpy arrays with shape (attn_length). These are
                the attention distributions so far.
            p_gens:
                List, same length as tokens, of floats, or None if not using pointer-generator
                model. The values of the generation probability so far.
            coverage:
                Numpy array of shape (attn_length), or None if not using coverage. The current
                coverage vector.
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
        """
        Return a NEW hypothesis, extended with the information from the latest step of beam search.
    
        Args:
            token:
                Integer. Latest token produced by beam search.
            token_string:
                string. Latest string produced by beam search.
            log_prob:
                Float. Log prob of the latest token.
            state:
                Current decoder state, a LSTMStateTuple.
            attn_dist:
                Attention distribution from latest step. Numpy array shape (attn_length).
            p_gen:
                Generation probability on latest step. Float.
            coverage:
                Latest coverage vector. Numpy array shape (attn_length), or None if not using
                coverage.
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
        """
        Determines if the current tokens make up a malformed hypothesis.
        """
        if _has_unknown_token(self.tokens, stop_token_id):
            return True
        if language_check.has_repeated_n_gram(self.tokens):
            return True
        if _has_repeated_entity(self.tokens, self.token_strings, vocab_size, comma_id):
            return True
        if language_check.sent_end_on_bad_word(self.token_strings):
            return True
        return False


    def score(self, vocab_size, key_token_ids, is_complete):
        """
        Returns a score for the hypothesis. If it breaks certain common-sense rules, return
        -10 ** 6. Else, return something like the mean log probability of a token in the sequence.
        
        Args:
            vocab_size: Integer.
            key_token_ids: dictionary containing IDs for 'stop', 'comma', 'period', 'pronouns'.
            is_complete: Whether to treat this as a complete output or partial output.
        """
        if is_complete in self._scores:
            # Score is already computed.
            return self._scores[is_complete]

        total_score = 0.

        if self._is_early_malformed(vocab_size, key_token_ids['stop'], key_token_ids['comma']):
            total_score -= 10. ** 6
        if is_complete and language_check.has_poor_grammar(self.token_strings):
            total_score -= 10. ** 6

        # Discourage using pronouns
        total_score -= sum(float(token in key_token_ids['pronouns']) for token in self.tokens)

        # Compute log probabilities
        log_probs = np.array(self.log_probs[1:])
        weights = np.ones_like(log_probs)

        assert len(set(len(array) for array in (log_probs, weights, self.attn_dists, self.p_gens))) == 1
        index_weights = [1., .9, .7]
        # Weigh the first three tokens more.
        for i, (index_weight, log_prob, attn_dist, p_gen) in enumerate(zip(
            index_weights, log_probs, self.attn_dists, self.p_gens
        )):
            max_attn = max(attn_dist)
            additional_log_prob_weight = index_weight * (1. - max_attn)
            log_probs[i] *= 1. + additional_log_prob_weight
            weights[i] += additional_log_prob_weight

        prob_score = weights.dot(log_probs) / weights.sum()
        total_score += prob_score

        # Add to score for being abstractive.
        total_score += .25 * np.mean([
            p if st != '.' else 0. for p, st in zip(self.p_gens, self.token_strings[1:])
        ])

        # Save computed score
        self._scores[is_complete] = total_score
        return total_score


def get_key_token_ids(vocab):
    return {
        'stop': vocab.word2id(data.STOP_DECODING, None),
        'comma': vocab.word2id(',', None),
        'period': vocab.word2id('.', None),
        'pronouns': {vocab.word2id(word, None) for word in ('he', 'she', 'him', 'her')},
        #'people': {vocab.word2id(word, None) for word in data.PERSON_TOKENS + ('[ORG]',)},
        #'other_entities': {
        #    vocab.word2id(word, None)
        #    for word in data.ENTITY_TOKENS + ('[PROPN]',)
        #    if word not in data.PERSON_TOKENS and word != '[ORG]'
        #},
    }


def run_beam_search(
    sess, model, vocab, batch, beam_size, max_dec_steps, min_dec_steps, trace_path=''
):
    """
    Performs beam search decoding on the given example.
  
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

    # Initialize beam_size-many hypotheses
    hyps = [
        Hypothesis(
            tokens=[vocab.word2id(data.START_DECODING, None)],
            token_strings=[data.START_DECODING],
            log_probs=[0.],
            state=dec_in_state,
            attn_dists=[],
            p_gens=[],
            # zero vector of length attention_length
            coverage=np.zeros([batch.enc_batch.shape[1]])
        )
        for _ in xrange(beam_size)
    ]
    # this will contain finished hypotheses (those that have emitted the [STOP] token)
    results = []
    key_token_ids = get_key_token_ids(vocab)

    steps = 0
    while steps < max_dec_steps and len(results) < 4 * beam_size:
        latest_tokens = [h.latest_token for h in hyps]
        # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        latest_tokens = [article_id_to_word_id.get(t, t) for t in latest_tokens]
        # list of current decoder states of the hypotheses
        states = [h.state for h in hyps]
        # list of coverage vectors (or None)
        prev_coverage = [h.coverage for h in hyps]

        # Run one step of the decoder to get the new info
        topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage = (
            model.decode_onestep(
                sess=sess,
                batch=batch,
                latest_tokens=latest_tokens,
                enc_states=enc_states,
                dec_init_states=states,
                prev_coverage=prev_coverage,
            )
        )

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        # On the first step, we only had one original hypothesis (the initial hypothesis). On
        # subsequent steps, all original hypotheses are distinct.
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        for i in xrange(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = (
                hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]
            )
            for j in xrange(2 * beam_size):
                # For each of the top 2 * beam_size hyps:
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
        # will contain hypotheses for the next step
        hyps = []
        for h in sort_hyps(all_hyps, vocab.size, key_token_ids, complete_hyps=False):
            # in order of most likely h
            if h.latest_token == vocab.word2id(data.STOP_DECODING, None):
                # Stop token is reached. If this hypothesis is sufficiently long, put in results.
                # Otherwise discard.
                if steps >= min_dec_steps:
                    results.append(h)
            elif h.latest_token >= data.N_FREE_TOKENS:
                # Hasn't reached stop token and generated non-unk token, so continue to extend this
                # hypothesis.
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == 4 * beam_size:
                # Once we've collected beam_size-many hypotheses for the next step, or
                # 4 * beam_size-many complete hypotheses, stop.
                break

        steps += 1

    if trace_path:
        for i, trace in enumerate(model._traces):
            with open(os.path.join(trace_path, 'timeline_%d.json' % i), 'w') as f:
                f.write(trace)

    # At this point, either we've got 4 * beam_size results, or we've reached maximum decoder steps

    if len(results) == 0:
        # if we don't have any complete results, add all current hypotheses (incomplete summaries)
        # to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results, vocab.size, key_token_ids, complete_hyps=True)

    # Return the hypothesis with highest average log prob
    best_hyp = hyps_sorted[0]
    score = best_hyp.score(vocab.size, key_token_ids, is_complete=True)
    return best_hyp, score


def sort_hyps(hyps, vocab_size, key_token_ids, complete_hyps):
    """
    Return a list of Hypothesis objects, sorted by descending average log probability.
    """
    return sorted(
        hyps,
        key=lambda h: h.score(vocab_size, key_token_ids, complete_hyps),
        reverse=True,
    )


def _has_unknown_token(tokens, stop_token_id):
    """
    Returns whether any of the tokens generated are unknown (except the STOP token).
    """
    if any(token < data.N_FREE_TOKENS for token in tokens[1:-1]):
        return True
    latest_token = tokens[-1]
    if latest_token < data.N_FREE_TOKENS and latest_token != stop_token_id:
        return True

    return False


def _has_repeated_entity(tokens, token_strings, vocab_size, comma_id):
    """
    Returns whether the same entity token is repeated back to back or separated by a comma.
    E.g. 'Obama Obama' or 'India, India'.
    """
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
