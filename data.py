"""
This file contains code to read the train/eval/test data from file and process it, and read the
vocab data from file and process it.
"""

import csv
import glob
import random
import re
import string
import struct
from collections import defaultdict
from tensorflow.core.example import example_pb2

PEOPLE_ID_SIZE = 16

# This is used to pad the encoder input, decoder input, and target sequence.
PAD_TOKEN = '[PAD]'
# This is used at the start of every decoder input sequence.
START_DECODING = '[START]'
# This is used at the end of untruncated target sequences.
STOP_DECODING = '[STOP]'

# These are used for tokens labeled as people by our people resolver. [PERSON_0] is the most
# commonly appearing person, all the way down to [PERSON_{PEOPLE_ID_SIZE}]. [PERSON] is for all
# remaining people tokens.
PERSON_TOKENS = tuple('[PERSON_%d]' % i for i in range(PEOPLE_ID_SIZE)) + ('[PERSON]',)
# These are all the entity tokens labeled by spacy (and includes people tokens above)
ENTITY_TOKENS = PERSON_TOKENS + (
    '[NORP]',
    '[FACILITY]',
    '[ORG]',
    '[GPE]',
    '[LOC]',
    '[PRODUCT]',
    '[EVENT]',
    '[WORK_OF_ART]',
    '[LANGUAGE]',
    '[DATE]',
    '[TIME]',
    '[PERCENT]',
    '[MONEY]',
    '[QUANTITY]',
    '[ORDINAL]',
    '[CARDINAL]',
)
# Part of speech tokens for out-of-vocab words.
POS_TOKENS = (
    '[ADJ]',
    '[ADP]',
    '[ADV]',
    '[CONJ]',
    '[DET]',
    '[INTJ]',
    '[NOUN]',
    '[NUM]',
    '[PART]',
    '[PRON]',
    '[PROPN]',
    '[PUNCT]',
    '[SYM]',
    '[VERB]',
    '[X]',
)
# Any out-of-vocab word that does not belong above.
UNKNOWN_TOKEN = '[UNK]'

UNKNOWN_TOKENS = ENTITY_TOKENS + POS_TOKENS + (UNKNOWN_TOKEN,)

N_IMPORTANT_TOKENS = len(ENTITY_TOKENS) + 3
N_FREE_TOKENS = len(UNKNOWN_TOKENS) + 3


class Vocab(object):
    """
    Vocabulary class for mapping between words and ids (integers)
    """

    def __init__(self, vocab_file, max_size):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0,
        reads the entire vocab file.
    
        Args:
            vocab_file:
                path to the vocab file, which is assumed to contain a word on each line,
                sorted with most frequent word first.
            max_size:
                integer. The maximum size of the resulting Vocabulary.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        # keeps track of total number of words in the Vocab
        self._count = 0

        # [PAD], [START], [STOP], and the UNKNOWN_TOKENS get the ids 0, 1, 2, 3...
        for w in (PAD_TOKEN, START_DECODING, STOP_DECODING) + UNKNOWN_TOKENS:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
        assert self._count == N_FREE_TOKENS

        allowed_chars = set(string.ascii_letters + string.punctuation)
        ascii_plus_period = set(string.ascii_letters + '.')

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.rstrip().split()
                if len(pieces) > 2:
                    print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
                    continue
                w = pieces[0]
                if w in (PAD_TOKEN, START_DECODING, STOP_DECODING) + UNKNOWN_TOKENS:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab '
                        'file, but %s is' % w
                    )
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                if any(c in w for c in ('[', ']', '{', '}')):
                    # these are used to mark word types and person ids.
                    continue
                if any(c not in allowed_chars for c in w):
                    continue
                if sum(1 for c in w if c not in ascii_plus_period) > 2:
                    continue

                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    break

        if max_size != 0 and self._count < max_size:
            raise Exception(
                'Could not read full vocab of size %d, only %d words found' % (
                    max_size, self._count
                )
            )


    def word2id(self, word, word_type):
        """
        Returns the id (integer) of a word (string) and word_type (string) pair. word_type can be
        a member of
            
        ENTITY_TOKENS: overrides the word in all cases
        POS_TOKENS: used if word is out-of-vocab
        """
        if word_type in ENTITY_TOKENS:
            return self._word_to_id[word_type]
        if word in self._word_to_id:
            return self._word_to_id[word]
        if word_type in POS_TOKENS:
            return self._word_to_id[word_type]
        return self._word_to_id[UNKNOWN_TOKEN]


    def id2word(self, word_id):
        """
        Returns the word (string) corresponding to an id (integer).
        """
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]


    @property
    def size(self):
        """
        Returns the total size of the vocabulary.
        """
        return self._count


    def write_metadata(self, fpath):
        """
        Writes metadata file for Tensorboard word embedding visualizer as described here:
        https://www.tensorflow.org/get_started/embedding_viz
    
        Args:
            fpath: place to write the metadata file
        """
        print "Writing word embedding metadata file to %s..." % (fpath)
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    """
    Generates tf.Examples from data files.
  
        Binary data format: <length><blob>. <length> represents the byte size
        of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
        the tokenized article text and summary.
  
    Args:
        data_path:
            Path to tf.Example data files. Can include wildcards, e.g. if you have several
            training data chunk files train_001.bin, train_002.bin, etc, then pass
            data_path=train_* to access them all.
        single_pass:
            Boolean. If True, go through the dataset exactly once, generating examples in the
            order they appear, then return. Otherwise, generate random examples indefinitely.
  
    Yields:
        Deserialized tf.Example.
    """
    while True:
        # get the list of datafiles
        filelist = glob.glob(data_path)
        assert filelist, ('Error: Empty filelist at %s' % data_path)
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    # finished reading this file
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print "example_generator completed reading all datafiles. No more data."
            break


def article2ids(article_words, vocab, copy_only_entities):
    """
    Map the article words to their ids. Also return a list of OOVs in the article.
  
    Args:
        article_words:
            list of word (string, word_type) tuples
        vocab:
            Vocabulary object
        copy_only_entities:
            boolean for whether non-entities can be copied
  
    Returns:
        ids:
            A list of word ids (integers); OOVs are represented by their temporary article OOV
            number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary
            OOV numbers will be 50000, 50001, 50002.
        oovs:
            A list of the OOV words in the article (strings), in the order corresponding to their
            temporary article OOV numbers.
        article_id_to_word_id:
            A map of temporary article OOV word id to vocab word id. This allows us to convert
            output ids back into an input id. As the same OOV word may have different UNK tokens,
            this maps to the majority for that token.
    """
    ids = []
    oovs = []
    unk_article_id_to_word_id_list = defaultdict(list) # for OOV ids
    unk_ids = set(vocab.word2id('', token) for token in UNKNOWN_TOKENS)

    for index, (w, word_type) in enumerate(article_words):
        i = vocab.word2id(w, word_type)
        if i in unk_ids and (not copy_only_entities or 3 <= i < N_IMPORTANT_TOKENS):
            if w not in oovs:
                # Add to list of OOVs
                oovs.append(w)
            # oov_num is 0 for the first article OOV, 1 for the second article OOV...
            oov_num = oovs.index(w)
            # id is e.g. 50000 for the first article OOV, 50001 for the second...
            ids.append(vocab.size + oov_num)
            unk_article_id_to_word_id_list[ids[-1]].append(i)
        else:
            ids.append(i)

    unk_article_id_to_word_id = {}
    # For every labeled OOV word, find the most common vocab word id.
    for article_id, word_ids in unk_article_id_to_word_id_list.iteritems():
        word_id_counts = defaultdict(int)
        # compute word_id count for each occurrence of article_id.
        for word_id in word_ids:
            word_id_counts[word_id] += 1
        # take most commonly labeled vocab word.
        sorted_words = sorted(word_id_counts.items(), key=lambda pair: pair[1], reverse=True)
        top_word_id = sorted_words[0][0]
        unk_article_id_to_word_id[article_id] = top_word_id

    return ids, oovs, unk_article_id_to_word_id


def abstract2ids(abstract_words, vocab, article_oovs, copyable_words, output_vocab_size):
    """
    Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
  
    Args:
        abstract_words:
            list of (word (string), word_type) tuples
        vocab:
            Vocabulary object
        article_oovs:
            list of in-article OOV words (strings), in the order corresponding to their
            temporary article OOV numbers
        copyable_words:
            set of all article words that can be copied
        output_vocab_size:
            int representing number of words that can be generated
  
    Returns:
        ids:
            List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers.
            Out-of-article OOV words are mapped to the UNK token id.
    """
    ids = []
    unk_ids = set(vocab.word2id('', token) for token in UNKNOWN_TOKENS)

    for w, word_type in abstract_words:
        # index ignoring entity / POS tags
        i_orig = vocab.word2id(w, None)
        # index including entity / POS tags
        i_real = vocab.word2id(w, word_type)
        # index if word is in article oov or entity
        i_article_oov = vocab.size + article_oovs.index(w) if w in article_oovs else 0
        # index for part of speech
        i_pos = vocab.word2id('', word_type)
        is_copyable = w in copyable_words

        if i_orig < output_vocab_size:
            # can be generated
            if i_real in unk_ids and i_article_oov:
                # labeled as an entity in both article and abstract
                ids.append(i_article_oov)
            else:
                ids.append(i_orig)
        elif i_orig not in unk_ids:
            # in vocab but cannot be generated
            if i_real in unk_ids and i_article_oov:
                # labeled as an entity in both article and abstract
                ids.append(i_article_oov)
            elif is_copyable:
                ids.append(i_orig)
            else:
                # can't be generated or copied, use POS
                ids.append(i_pos)
        else:
            # out-of-vocab
            if is_copyable:
                assert i_article_oov
                ids.append(i_article_oov)
            else:
                # can't be generated or copied, use POS
                ids.append(i_pos)

    return ids


def outputid_to_word(id_, vocab, article_oovs):
    """
    Maps output ids to words, including mapping in-article OOVs from their temporary ids to the
    original OOV string (applicable in pointer-generator mode).
  
    Args:
        id_: integer
        vocab: Vocabulary object
        article_oovs: list of OOV words (strings) in the order corresponding to their temporary
            article OOV ids
  
    Returns:
        word: string
    """
    try:
        # might be [UNK]
        w = vocab.id2word(id_)
    except ValueError:
        # w is OOV
        article_oov_idx = id_ - vocab.size
        w = article_oovs[article_oov_idx]

    return w


def show_art_oovs(article, vocab):
    """
    Returns the article string, highlighting the OOVs by placing __underscores__ around them.
    """
    unk_ids = set(vocab.word2id('', token) for token in UNKNOWN_TOKENS)
    words = [parse_word(word) for word in article.split(' ')]
    words = [
        ("__%s__" % w) if vocab.word2id(w, word_type) in unk_ids else w
        for w, word_type in words
    ]

    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    """
    Returns the abstract string, highlighting the article OOVs with __underscores__.
    Non-article OOVs are differentiated like !!__this__!!.
  
    Args:
        abstract: string
        vocab: Vocabulary object
        article_oovs: list of words (strings)
    """
    unk_ids = set(vocab.word2id('', token) for token in UNKNOWN_TOKENS)
    words = [parse_word(word) for word in abstract.split(' ')]
    new_words = []

    for w, word_type in words:
        if vocab.word2id(w, word_type) in unk_ids:
            # w is oov
            if w in article_oovs:
                # word appeared in article
                new_words.append("__%s__" % w)
            elif vocab.word2id(w, None) in unk_ids:
                # word is unknown and does not appear in article
                new_words.append("!!__%s__!!" % w)
            else:
                # word is known but was labeled as entity
                new_words.append(w)
        else:
            # w is in-vocab word
            new_words.append(w)

    out_str = ' '.join(new_words)
    return out_str


def parse_word(word):
    """
    Returns (word, word_type).
     
    word can be of the form:
    
    - "word{i}" -> word, PERSON_i
    - "word[POS]" -> word, POS
    - "word" -> word, None
    """
    def find_match(pattern):
        match = re.search(pattern, word)
        if match:
            return word[:match.start()], word[match.start(): match.end()]
        return word, ''

    real_word, person_id = find_match(r'(\{.*\})')
    if person_id:
        # has person id tag
        person_id = int(person_id[1: -1])
        if person_id < PEOPLE_ID_SIZE:
            return real_word, PERSON_TOKENS[person_id]
        else:
            # person id is too large, return generic person token
            return real_word, PERSON_TOKENS[-1]

    real_word, word_type = find_match(r'(\[.*\])')
    if word_type:
        if word_type in ENTITY_TOKENS + POS_TOKENS:
            return real_word, word_type
        else:
            return real_word, None

    return word, None
