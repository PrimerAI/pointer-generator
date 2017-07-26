"""
Library method to summarize given text.
"""
import os


_model_dir = 'saved_models/combined_test'
_vocab_path = os.path.join(_model_dir, 'vocab')
_vocab_size = 20000
_beam_size = 5

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
        trace_path='',
    )
    _hps = Hps(
        # parameters important for decoding
        batch_size=_beam_size,
        copy_only_entities=False,
        coverage=False,
        emb_dim=128,
        enc_hidden_dim=256,
        dec_hidden_dim=400,
        max_enc_steps=500,
        mode='decode',
        output_vocab_size=20000,
        restrictive_embeddings=False,
        save_matmul=False,
        tied_output=True,
        two_layer_encoder=False,
        # other parameters
        adagrad_init_acc=.1,
        adam_optimizer=True,
        cov_loss_wt=1.,
        lr=.15,
        max_dec_steps=1,
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
    config = tf.ConfigProto(allow_soft_placement=True)
    _sess = tf.Session(config=config)
    ckpt_state = tf.train.get_checkpoint_state(_model_dir)
    saver.restore(_sess, ckpt_state.model_checkpoint_path)


def generate_summary(spacy_article, ideal_summary_length_tokens=60):
    """
    Input: spacy-processed text. Should be the output of doc.spacy_text().
    
    Output: unicode summary of the text and scalar score of its quality.
    """
    # These imports are slow - lazy import.
    from batcher import Batch, Example
    from beam_search import run_beam_search
    from io_processing import process_article, process_output

    if _model is None:
        _load_model()

    # Handle short inputs
    article_tokens, _, orig_article_tokens = process_article(spacy_article)
    if len(article_tokens) <= ideal_summary_length_tokens:
        return spacy_article.text
    min_summary_length = min(10 + len(article_tokens) / 10, 2 * ideal_summary_length_tokens / 3)
    max_summary_length = min(10 + len(article_tokens) / 5, 3 * ideal_summary_length_tokens / 2)

    # Make input data
    example = Example(' '.join(article_tokens), abstract='', vocab=_vocab, hps=_hps)
    batch = Batch([example] * _beam_size, _hps, _vocab)

    # Generate output
    hyp, score = run_beam_search(
        _sess, _model, _vocab, batch, _beam_size, max_summary_length, min_summary_length
    )

    # Extract the output ids from the hypothesis and convert back to words
    return process_output(hyp.token_strings[1:], orig_article_tokens), hyp, score

