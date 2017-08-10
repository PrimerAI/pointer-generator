"""
This file contains code to run beam search decoding.
"""

import numpy as np
import os

import data
import language_check


class Hypothesis(object):
    """
    Class to represent a hypothesis (partial output of a summary) during beam search. Holds all
    the information needed for the hypothesis.
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
                Current state of the decoder, a LSTMStateTuple. If two layer LSTM, then a tuple of
                LSTMStateTuples.
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


    def extend(self, token, token_string, log_prob, state, attn_dist, p_gen, coverage):
        """
        Return a NEW hypothesis, extended with the information from the latest step of beam search.
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
        -10 ** 6. Else, return the average log probability of a token in the sequence with
        modifications for using pronouns and generation probabilities.
        
        Args:
            vocab_size: Integer.
            key_token_ids: dictionary containing IDs for 'stop', 'comma', 'period', 'pronouns'.
            is_complete: Whether to treat this as a complete output or partial output.
        
        Returns:
            total_score: Float.
        """
        total_score = 0.

        if self._is_early_malformed(vocab_size, key_token_ids['stop'], key_token_ids['comma']):
            total_score -= 10. ** 6
        if is_complete and language_check.has_poor_grammar(self.token_strings):
            total_score -= 10. ** 6

        # Discourage using pronouns
        total_score -= .05 * sum(float(token in key_token_ids['pronouns']) for token in self.tokens)

        # Compute log probabilities
        total_score += sum(self.log_probs[1:]) / (len(self.log_probs) - 1)

        # Abstractive tokens tend to have lower log probabilities, so compensate for that.
        total_score += .25 * np.mean([
            p if st != '.' else 0. for p, st in zip(self.p_gens, self.token_strings[1:])
        ])

        # Encourage entities among the first 15 tokens
        people_score = 0.
        org_score = 0.
        for i, token in enumerate(self.tokens[:15]):
            if token in key_token_ids['people']:
                people_score = max(people_score, 1. - i / 15.)
            elif token in key_token_ids['orgs']:
                org_score = max(org_score, 1. - i / 15.)

        total_score += max(.15 * people_score, .1 * org_score)

        return total_score


    @property
    def mean_llh(self):
        """
        Computes mean log-likelihood of each output token.
        """
        return sum(self.log_probs[1:]) / (len(self.log_probs) - 1)


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
        beam_size: Integer, size of the search at each step
        max_dec_steps: Integer, stop search after this many steps
        min_dec_steps: Integer, accept results of at least this length only
        trace_path: string, if provided save trace results to this path
  
    Returns:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
        score: the score of the best hypothesis.
    """
    # Run the encoder to get the encoder hidden states and decoder initial state.
    # enc_states has shape [batch_size, <=max_enc_steps, 2*enc_hidden_dim].
    # dec_in_state is a LSTMStateTuple, or if two layer lstm then a tuple of LSTMStateTuples.
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
            coverage=np.zeros([batch.enc_batch.shape[1]]),
        )
        for _ in xrange(beam_size)
    ]
    # This will contain finished hypotheses (those that have emitted the [STOP] token).
    results = []
    # Ids for tokens that will be needed for scoring hypotheses.
    org_id = vocab.word2id('[ORG]', None)
    key_token_ids = {
        'stop': vocab.word2id(data.STOP_DECODING, None),
        'comma': vocab.word2id(',', None),
        'period': vocab.word2id('.', None),
        'pronouns': {vocab.word2id(word, None) for word in ('he', 'she', 'him', 'her')},
        'people': set(
            article_id for article_id, word_id in batch.article_id_to_word_ids[0].iteritems()
            if 3 <= word_id < len(data.PERSON_TOKENS) + 3
        ),
        'orgs': set(
            article_id for article_id, word_id in batch.article_id_to_word_ids[0].iteritems()
            if word_id == org_id
        ),
    }

    steps = 0
    while steps < max_dec_steps and len(results) < 4 * beam_size:
        latest_tokens = [h.latest_token for h in hyps]
        # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        latest_tokens = [batch.article_id_to_word_ids[0].get(t, t) for t in latest_tokens]
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
                token_string = data.outputid_to_word(topk_ids[i, j], vocab, batch.art_oovs[0])
                # For each of the top 2 * beam_size hyps:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(
                    token=topk_ids[i, j],
                    token_string=token_string,
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
        # If needed, record trace of the search performance.
        for i, trace in enumerate(model._traces):
            with open(os.path.join(trace_path, 'timeline_%d.json' % i), 'w') as f:
                f.write(trace)

    # At this point, either we've got 4 * beam_size results, or we've reached maximum decoder steps.

    if len(results) == 0:
        # If we don't have any complete results, add all current hypotheses (incomplete summaries)
        # to results. Note: we still use complete_hyps=True in the next step since we want to
        # check for valid grammar properties.
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
    Returns whether any of the tokens generated are unknown (except possible a final STOP token).
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
