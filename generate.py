import os
import sys


_model_dir = 'saved_model'
_vocab_path = os.path.join(_model_dir, 'vocab')
_vocab_size = 20000
_beam_size = 4

_settings = None
_hps = None
_vocab = None
_sess = None
_model = None


def _load_model():
    import tensorflow as tf
    from data import Vocab
    from model import Hps, Settings, SummarizationModel

    global _settings, _hps, _vocab, _sess, _model

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

    _vocab = Vocab(_vocab_path, _vocab_size)
    _model = SummarizationModel(_settings, _hps, _vocab)
    _model.build_graph()

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    _sess = tf.Session(config=config)
    ckpt_state = tf.train.get_checkpoint_state(_model_dir)
    saver.restore(_sess, ckpt_state.model_checkpoint_path)


def generate_summary(article, ideal_summary_length_tokens=60):
    from batcher import Batch, Example
    from beam_search import run_beam_search
    from io_processing import process_article, process_output

    if _model is None:
        _load_model()

    if not isinstance(article, unicode):
        article = unicode(article, 'utf-8')

    article_tokens, orig_article_tokens = process_article(article)
    if len(article_tokens) <= ideal_summary_length_tokens:
        return article
    min_summary_length = min(10 + len(article_tokens) / 10, 2 * ideal_summary_length_tokens / 3)
    max_summary_length = min(10 + len(article_tokens) / 5, 3 * ideal_summary_length_tokens / 2)

    # make input data
    example = Example(' '.join(article_tokens), abstract='', vocab=_vocab, hps=_hps)
    batch = Batch([example] * _beam_size, _hps, _vocab)

    # generate output
    best_hyp, best_score = run_beam_search(
        _sess, _model, _vocab, batch, _beam_size, max_summary_length, min_summary_length
    )

    # Extract the output ids from the hypothesis and convert back to words
    decoded_tokens = best_hyp.tokens[1:]
    return process_output(
        decoded_tokens, orig_article_tokens, example.article_id_to_word_index, _vocab
    )

