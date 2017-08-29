import json
from pytest import raises

from decoder import generate_summary
from primer_core.analytic_pipelines.base.document_pipeline import SingleDocument
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
    Test that the summary of a real article is the same as generated during a local test.
    Note: this test is slow (20 seconds not including imports / loading model).
    """
    # load data
    with open('test_article.json') as f:
        data = json.load(f)

    # compute summary
    doc = SingleDocument(document_id=0, raw={'body': data['article']})
    summary, score = generate_summary(doc.spacy_text())

    # check result
    assert isinstance(summary, unicode)
    assert summary == data['expected_summary']
    assert abs(score - data['expected_score']) < .001

