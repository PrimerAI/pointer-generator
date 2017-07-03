import numpy as np
import os
import spacy
import string
import struct
import sys
from sklearn.decomposition.truncated_svd import TruncatedSVD
from tensorflow.core.example import example_pb2

from data import N_FREE_TOKENS, Vocab

def compute_reduced_embeddings_original_vocab(
    output_vocab_filepath, output_embeddings_filepath, input_vocab_filepath, vocab_size,
    embedding_dim
):
    print N_FREE_TOKENS
    vocab = Vocab(input_vocab_filepath, 1.5 * vocab_size)
    spacy_vocab = spacy.load('en').vocab
    matrix = np.zeros((vocab_size, spacy_vocab.vectors_length), dtype=np.float32)
    new_i = 0
    final_vocab = []

    for i, word in vocab._id_to_word.iteritems():
        if new_i == vocab_size:
            break

        vector = spacy_vocab[unicode(word)].vector
        if i >= N_FREE_TOKENS and np.allclose(vector, 0.):
            continue

        if i >= N_FREE_TOKENS:
            final_vocab.append(word)
        matrix[new_i] = vector
        new_i += 1

    if embedding_dim < spacy_vocab.vectors_length:
        svd = TruncatedSVD(n_components=embedding_dim, algorithm='arpack')
        embeddings = svd.fit_transform(matrix)
        print embeddings.shape
        print [sum(svd.explained_variance_ratio_[:i]) for i in range(1, embedding_dim + 1)]
    else:
        embeddings = matrix

    with open(output_vocab_filepath, 'w') as output:
        for word in final_vocab:
            output.write('%s\n' % word)
    np.save(output_embeddings_filepath, embeddings)


def write_dummy_example(out_file):
    def write_single(article, abstract):
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article])
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)

        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

    with open(out_file, 'wb') as writer:
        for i in range(1000):
            article = 'hi there Michael{1} . that was Jake{2} .'
            abstract = '<s> bye Michael{1} guy . </s>'
            write_single(article, abstract)

            article = 'hi there Michael{1} . this is Jake{2} .'
            abstract = '<s> bye Jake{2} . </s>'
            write_single(article, abstract)

"""
def write_spacy_vocab(output_dirpath, vocab_size, embedding_dim):
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    allowed_chars = set(string.ascii_letters + string.punctuation)
    ascii = set(string.ascii_letters)
    ascii_plus_period = set(string.ascii_letters + '.')
    word_set = set()
    spacy_vocab = spacy.load('en').vocab
    top_words = []

    for w in spacy_vocab:
        if w.rank > 2 * vocab_size:
            continue
        try:
            word_string = str(w.lower_).strip()
            if not word_string:
                continue
            if word_string in word_set:
                continue
            if any(bad_char in word_string for bad_char in ('[', ']', '<', '>', '{', '}')):
                # these are used to mark word types and person ids.
                continue
            if any(c not in allowed_chars for c in word_string):
                continue
            if sum(1 for c in word_string if c not in ascii_plus_period) > 2:
                continue
            if word_string[-1] == '.' and sum(1 for c in word_string if c in ascii) > 2:
                continue

            top_words.append(w)
            word_set.add(word_string)
        except:
            pass

    top_words.sort(key=lambda w: w.rank)
    top_words = top_words[:vocab_size]

    with open(os.path.join(output_dirpath, 'vocab'), 'w') as f:
        for word in top_words:
            f.write('%s\n' % word.lower_.strip())

    vectors = np.array([w.vector for w in top_words])
    svd = TruncatedSVD(n_components=embedding_dim, algorithm='arpack')
    embeddings = svd.fit_transform(vectors)

    print embeddings.shape
    print [sum(svd.explained_variance_ratio_[:i]) for i in range(1, embedding_dim + 1)]
    np.save(os.path.join(output_dirpath, 'pretrained_embeddings.npy'), embeddings)
"""


def compute_vocab_overlap(vocab_1, vocab_2):
    matches = [[0] * 5 for i in range(5)]
    missing = []

    for w1, r1 in vocab_1.iteritems():
        r2 = vocab_2.get(w1, 100000)
        for i1, rank1 in enumerate((10000, 20000, 30000, 40000, 50000)):
            if r1 >= rank1:
                continue

            for i2, rank2 in enumerate((10000, 20000, 30000, 40000, 50000)):
                if r2 >= rank2:
                    if i1 == 0 and i2 == 1:
                        missing.append((w1, r1))
                    continue

                matches[i1][i2] += 1

    for row in matches:
        print row

    missing.sort(key=lambda pair: pair[1])
    for item in missing:
        print item

def read_vocab(filename):
    vocab = {}

    with open(filename) as f:
        for i, line in enumerate(f):
            word = line.split()[0]
            vocab[word] = i

            if i == 100:
                print vocab

    return vocab

def see_vocab_overlap(filepath1, filepath2):
    vocab1 = read_vocab(filepath1)
    vocab2 = read_vocab(filepath2)
    compute_vocab_overlap(vocab1, vocab2)

if __name__ == '__main__':
    #assert len(sys.argv) == 4
    #write_dummy_example(sys.argv[1])
    #see_vocab_overlap(sys.argv[1], sys.argv[2])
    compute_reduced_embeddings_original_vocab(sys.argv[1], sys.argv[2], sys.argv[3], 20000, 300)
