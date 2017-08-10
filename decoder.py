"""
Seq-to-seq based summarization method. Model is based off of
https://github.com/abisee/pointer-generator and is trained on 300K news articles from
CNN / Dailymail and 100K new cables.
"""
import os
from spacy.tokens.doc import Doc


_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_parameters')
_vocab_path = os.path.join(_model_dir, 'vocab')
_vocab_size = 20000
_beam_size = 4

_settings = None
_hps = None
_vocab = None
_sess = None
_model = None


def _load_model():
    # These imports are slow - lazy import.
    import tensorflow as tf
    from data import Vocab
    from model import Hps, Settings, SummarizationModel

    global _settings, _hps, _vocab, _sess, _model

    # Define settings and hyperparameters
    _settings = Settings(
        embeddings_path='',
        log_root='',
        trace_path='',# traces/traces_blog',
    )
    _hps = Hps(
        # parameters important for decoding
        batch_size=_beam_size,
        copy_only_entities=False,
        coverage=False,
        emb_dim=128,
        enc_hidden_dim=256,
        dec_hidden_dim=400,
        max_dec_steps=1,
        max_enc_steps=400,
        mode='decode',
        output_vocab_size=20000,
        restrictive_embeddings=False,
        save_matmul=False,
        tied_output=True,
        two_layer_lstm=False,
        # other parameters
        adagrad_init_acc=.1,
        adam_optimizer=True,
        cov_loss_wt=1.,
        high_attn_loss_wt=0.,
        lr=.15,
        max_grad_norm=2.,
        people_loss_wt=0.,
        rand_unif_init_mag=.02,
        trunc_norm_init_std=1e-4,
    )

    # Define model
    _vocab = Vocab(_vocab_path, _vocab_size)
    _model = SummarizationModel(_settings, _hps, _vocab)
    _model.build_graph()

    # Load model from disk
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        #intra_op_parallelism_threads=1,
        #inter_op_parallelism_threads=1,
    )
    _sess = tf.Session(config=config)
    ckpt_state = tf.train.get_checkpoint_state(_model_dir)
    saver.restore(_sess, ckpt_state.model_checkpoint_path)


def generate_summary(spacy_article, ideal_summary_length_tokens=60):
    """
    Generates summary of the given article. Note that this is slow (~20 seconds on a single CPU).
    
    Args:
        spacy_article: Spacy-processed text. The model was trained on the output of
        doc.spacy_text(), so for best results the input here should also come from doc.spacy_text().
    
    Returns:
        Tuple of unicode summary of the text and scalar score of its quality. Score is approximately
        an average log-likelihood of the summary (so it is < 0.) and typically is in the range
        [-.2, -.5]. Summaries with scores below -.4 are usually not very good.
    """
    assert isinstance(spacy_article, Doc)

    # These imports are slow - lazy import.
    import time
    from batcher import Batch, Example
    from beam_search import run_beam_search
    from io_processing import process_article, process_output

    if _model is None:
        _load_model()

    # Handle short inputs
    t0 = time.time()
    article_tokens, _, orig_article_tokens = process_article(spacy_article)
    print 'processing input', time.time() - t0
    if len(article_tokens) <= ideal_summary_length_tokens:
        return spacy_article.text, 0.

    min_summary_length = min(10 + len(article_tokens) / 10, 2 * ideal_summary_length_tokens / 3)
    max_summary_length = min(10 + len(article_tokens) / 5, 3 * ideal_summary_length_tokens / 2)

    # Make input data
    example = Example(' '.join(article_tokens), abstract='', vocab=_vocab, hps=_hps)
    batch = Batch([example] * _beam_size, _hps, _vocab)

    # Generate output
    t1 = time.time()
    hyp, score = run_beam_search(
        _sess, _model, _vocab, batch, _beam_size, max_summary_length, min_summary_length,
        _settings.trace_path,
    )
    print 'beam search', time.time() - t1

    # Extract the output ids from the hypothesis and convert back to words
    return process_output(hyp.token_strings[1:], orig_article_tokens), score
