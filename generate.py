import os
import sys
import tensorflow as tf

from batcher import Batch, Example
import beam_search
from data import Vocab
from io_processing import process_article, process_output
from model import Hps, Settings, SummarizationModel


_settings = Settings(
    embeddings_path='',
    log_root='',
    trace_path='',
)

_hps = Hps(
    adagrad_init_acc=.1,
    adam_optimizer=True,
    batch_size=4,
    cov_loss_wt=1.,
    coverage=True,
    emb_dim=128,
    enc_hidden_dim=256,
    dec_hidden_dim=400,
    lr=.15,
    max_dec_steps=1,
    max_enc_steps=400,
    max_grad_norm=2.,
    mode='decode',
    rand_unif_init_mag=.02,
    restrictive_embeddings=False,
    save_matmul=True,
    trunc_norm_init_std=1e-4,
)

_model_dir = 'saved_model'
_vocab_path = os.path.join(_model_dir, 'vocab')
_vocab_size = 20000
_beam_size = 4
_ideal_summary_length = 60


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, vocab, hps):
        """
        Initialize decoder.
        """
        self._model = model
        self._vocab = vocab
        self._hps = hps

        # load model
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        self._sess = tf.Session(config=config)
        ckpt_state = tf.train.get_checkpoint_state(_model_dir)
        saver.restore(self._sess, ckpt_state.model_checkpoint_path)


    def generate_summary(self, article):
        if not isinstance(article, unicode):
            article = unicode(article, 'utf-8')

        article_tokens, orig_article_tokens = process_article(article)
        if len(article_tokens) <= _ideal_summary_length:
            return article
        min_summary_length = min(10 + len(article_tokens) / 10, 2 * _ideal_summary_length / 3)
        max_summary_length = min(10 + len(article_tokens) / 5, 3 * _ideal_summary_length / 2)

        # make input data
        example = Example(' '.join(article_tokens), abstract='', vocab=self._vocab, hps=self._hps)
        batch = Batch([example] * _beam_size, self._hps, self._vocab)

        # generate output
        best_hyp, best_score = beam_search.run_beam_search(
            self._sess, self._model, self._vocab, batch, _beam_size, max_summary_length,
            min_summary_length
        )

        # Extract the output ids from the hypothesis and convert back to words
        decoded_tokens = best_hyp.tokens[1:]
        return process_output(
            decoded_tokens, orig_article_tokens, example.article_id_to_word_index, self._vocab
        )


_decoder = None

def get_decoder():
    global _decoder

    if _decoder is None:
        vocab = Vocab(_vocab_path, _vocab_size)
        model = SummarizationModel(_settings, _hps, vocab)
        model.build_graph()
        _decoder = BeamSearchDecoder(model, vocab, _hps)

    return _decoder


if __name__ == '__main__':
    decoder = get_decoder()
    f = open(sys.argv[1])
    out = open(sys.argv[2], 'w')
    for line in f:
        parts = line.rstrip().split('\t')
        summary = decoder.generate_summary(parts[0])
        out.write('\t'.join(parts + [summary.encode('utf-8')]) + '\n')
        out.flush()
    f.close()
    out.close()

