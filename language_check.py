
_articles_and_prepositions = {
    # articles
    'a',
    'an',
    'the',
    # prepositions
    'aboard',
    'about',
    'above',
    'across',
    'after',
    'against',
    'along',
    'amid',
    'among',
    'anti',
    'around',
    'as',
    'at',
    'before',
    'behind',
    'below',
    'beneath',
    'beside',
    'besides',
    'between',
    'beyond',
    'but',
    'by',
    'concerning',
    'considering',
    'despite',
    'down',
    'during',
    'except',
    'excepting',
    'excluding',
    'following',
    'for',
    'from',
    'in',
    'inside',
    'into',
    'like',
    'minus',
    'near',
    'of',
    'off',
    'on',
    'onto',
    'opposite',
    'outside',
    'over',
    'past',
    'per',
    'plus',
    'regarding',
    'round',
    'save',
    'since',
    'than',
    'through',
    'to',
    'toward',
    'towards',
    'under',
    'underneath',
    'unlike',
    'until',
    'up',
    'upon',
    'versus',
    'via',
    'with',
    'within',
    'without',
}


def has_repeated_n_gram(tokens, disallowed_n=3):
    """
    Returns whether the sequence has any n_grams repeated.
    """
    seen_n_grams = set()

    for i in range(len(tokens) - disallowed_n + 1):
        n_gram = tuple(tokens[i: i + disallowed_n])
        if n_gram in seen_n_grams:
            return True
        seen_n_grams.add(n_gram)

    return False


def sent_end_on_bad_word(token_strings):
    """
    Returns whether there is an article or preposition that precedes a period.
    """
    for i in range(len(token_strings) - 1):
        if token_strings[i] in _articles_and_prepositions and token_strings[i + 1] == '.':
            return True
    return False


def has_poor_grammar(token_strings):
    """
    Returns whether the output has an odd number of double quotes or if it does not have balanced
    parentheses.
    """
    has_open_left_parens = False
    quote_count = 0

    for token in token_strings:
        if token == '(':
            if has_open_left_parens:
                return True
            else:
                has_open_left_parens = True
        elif token == ')':
            if has_open_left_parens:
                has_open_left_parens = False
            else:
                return True
        elif token == '"':
            quote_count += 1

    return quote_count % 2 == 1 or has_open_left_parens
