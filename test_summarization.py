import json
from pytest import raises

from decoder import generate_summary
from primer_core.nlp.get_spacy import get_spacy


def test_incorrect_inputs():
    with raises(AssertionError):
        generate_summary(None)
    with raises(AssertionError):
        generate_summary(u'Random string')


def test_short_input():
    text_input = u'Short phrase.'
    summary, score = generate_summary(get_spacy()(text_input))
    assert summary == text_input
    assert score == 0.


def test_real_article():
    """
    Test that the summary of a real article is sensible. Note: this test is slow (20 seconds not
    including imports / loading model).
    """
    # load data
    with open('test_article.json') as f:
        data = json.load(f)

    # compute summary
    spacy_article = get_spacy()(data['text'])
    summary, score, llh = generate_summary(spacy_article)

    # check summary
    assert isinstance(summary, unicode)
    assert 250 < len(summary) < 500
    assert any(c.isupper() for c in summary)
    assert '.' in summary
    lower_case_summary = summary.lower()
    for word in data['required_words']:
        assert word.lower() in lower_case_summary

    # check score
    assert -5. < score < -.1
