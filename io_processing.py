import string
import unicodedata
from collections import defaultdict

import data
from nltk.tokenize.moses import MosesDetokenizer
from primer_core.entities.people.people_resolver import SpacyPeopleResolver


assert all(token[0] == '[' and token[-1] == ']' for token in data.ENTITY_TOKENS + data.POS_TOKENS)
ENTITY_TAGS = tuple(token[1: -1] for token in data.ENTITY_TOKENS)
POS_TAGS = tuple(token[1: -1] for token in data.POS_TOKENS)


def process_article(spacy_article, print_edge_cases=False):
    """
    Tags the tokens from spacy_article with the first of the following:
    
    {PERSON_X}:
        if the token is part of the Xth most important person, as determined by
        SpacyPeopleResolver
    [entity_type]:
        if the token is an entity as determined by spacy.
    [part_of_speech]:
        part of speech as determined by spacy.
    
    Return both the newly tagged tokens as well as the text of the original tokens.
    """
    span_to_person_id = _resolve_people(spacy_article)
    case_sensitive_article_tokens = []
    article_tokens = []
    artice_token_indices = []

    for token in spacy_article:
        orig_token_text = token.text.strip()
        orig_token_text = orig_token_text.replace('[', '(').replace(']', ')')
        orig_token_text = orig_token_text.replace('{', '(').replace('}', ')')
        if not orig_token_text:
            continue

        case_sensitive_article_tokens.append(orig_token_text)
        token_text = orig_token_text.lower()

        person_id = _find_person_span_and_update(
            spacy_article.text, span_to_person_id, token.idx, token.idx + len(token.text)
        )
        if person_id is not None:
            token_text += '{%d}' % person_id
        elif token.ent_type_ in ENTITY_TAGS:
            token_text += '[%s]' % token.ent_type_
        elif token.pos_ in POS_TAGS:
            token_text += '[%s]' % token.pos_

        article_tokens.append(token_text)
        artice_token_indices.append(token.idx)

    if print_edge_cases and span_to_person_id:
        print '################'
        print "Person mention not fully found:"
        print span_to_person_id

    return article_tokens, artice_token_indices, case_sensitive_article_tokens


def _resolve_people(spacy_article):
    """
    Run SpacyPeopleResolver with very low confidence thresholds (it's ok to label other entities as
    people for example).
    
    Label the found people from most popularly occurring to least popularly occurring, and return
    a map from each mention's span to the person id.
    """
    # run people resolver
    people_resolver = SpacyPeopleResolver(
        {0: [spacy_article]},
        min_num_persons=1,
        min_person_label_ratio=.1,
        min_p_entity=.1,
        min_p_person=.3,
        min_unambiguous_p=.5,
    )
    people_resolver.resolve(min_p=.5)

    # collect spans for each person_id
    person_to_span = defaultdict(list)
    for key, person_id in people_resolver.key_to_person_root_.iteritems():
        span = people_resolver.occurrences_[key][1][0]
        span = _strip_span(span, spacy_article.text[span[0]: span[1]])
        person_to_span[person_id].append(span)

    # sort people by count and then order of appearance (id 0 is most popular)
    spans_by_person = sorted(
        person_to_span.values(),
        key=lambda spans: 100 * len(spans) - min(spans)[0],
        reverse=True,
    )
    span_to_person_id = {span: i for i, spans in enumerate(spans_by_person) for span in spans}

    return span_to_person_id


def _find_person_span_and_update(text, span_to_person_id, start, end):
    """
    Try to find a person span containing the search span. If found, reduce the person span by
    removing the search span from it. Returns the person id of the found span.
    """
    span = _find_span(span_to_person_id, start, end)
    if span is None:
        return None

    person_id = span_to_person_id.pop(span)
    remaining_mention = text[end: span[1]].lstrip()
    if remaining_mention:
        span_to_person_id[(span[1] - len(remaining_mention), span[1])] = person_id

    return person_id


def _find_span(span_to_person_id, start, end):
    """
    Find an containing span of the search span.
    """
    for (span_start, span_end), person_id in span_to_person_id.iteritems():
        if start >= span_start and end <= span_end:
            return span_start, span_end
    return None


def _strip_span(span, text):
    """
    Return span after stripping out whitespace on either side of text.
    """
    start = span[0] + len(text) - len(text.lstrip())
    end = span[1] - (len(text) - len(text.rstrip()))
    return start, end


_moses_detokenizer = MosesDetokenizer()
_end_sentence_punc = {'.', '!', '?'}
_punctuation = set(string.punctuation)


def _is_punctuation(token):
    return (
        len(token) == 1 and
        (token in _punctuation or unicodedata.category(token).startswith('P'))
    )


def process_output(summary_token_strings, article_token_strings):
    summary_token_strings = _fix_word_capitalizations(summary_token_strings, article_token_strings)
    summary_token_strings = _fix_contractions(summary_token_strings)
    _fix_ending(summary_token_strings)
    _capitalize_start_sentence(summary_token_strings)
    merged_summary = _moses_detokenizer.detokenize(summary_token_strings, return_str=True)
    return merged_summary


def _fix_word_capitalizations(token_strings, article_token_strings):
    word_capitalizations = defaultdict(lambda: defaultdict(int))

    for i, word in enumerate(article_token_strings):
        if i > 0 and not _is_punctuation(article_token_strings[i - 1]):
            word_capitalizations[word.lower()][word] += 1

    best_word_capitalizations = {}
    for word_lower, cap_counts in word_capitalizations.iteritems():
        sorted_capitalizations = sorted(
            cap_counts.items(),
            key=lambda item: 100 * item[1] + _count_capital_letters(item[0]),
            reverse=True,
        )
        best_word_capitalizations[word_lower] = sorted_capitalizations[0][0]

    return [
        best_word_capitalizations.get(token_string, token_string)
        for token_string in token_strings
    ]


def _count_capital_letters(word):
    return sum(1 for c in word if c.isupper())


def _fix_contractions(token_strings):
    fixed_token_strings = []

    for i, token_string in enumerate(token_strings):
        if i > 0 and token_string == "n't":
            fixed_token_strings[-1] += "n't"
        else:
            fixed_token_strings.append(token_string)

    return fixed_token_strings


def _fix_ending(token_strings):
    if token_strings[-1] == data.STOP_DECODING:
        del token_strings[-1]
        if (
            token_strings[-1] not in _end_sentence_punc and
            token_strings[-2] not in _end_sentence_punc
        ):
            token_strings.append('.')
    else:
        token_strings.append('...')


def _capitalize_start_sentence(token_strings):
    is_new_sentence = True
    for i, token_string in enumerate(token_strings):
        if token_string in _end_sentence_punc:
            is_new_sentence = True
        elif is_new_sentence and token_string not in _punctuation:
            token_strings[i] = token_string.capitalize()
            is_new_sentence = False

