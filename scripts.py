import numpy as np
import os
import spacy
import string
import struct
import sys
from sklearn.decomposition.truncated_svd import TruncatedSVD
from tensorflow.core.example import example_pb2

from data import Vocab

def compute_reduced_embeddings_original_vocab(output_filepath, vocab_filepath, vocab_size, embedding_dim):
    vocab = Vocab(vocab_filepath, vocab_size)
    spacy_vocab = spacy.load('en').vocab
    matrix = np.zeros((vocab_size, spacy_vocab.vectors_length), dtype=np.float32)

    for i, word in vocab._id_to_word.iteritems():
        matrix[i] = spacy_vocab[unicode(word)].vector

    svd = TruncatedSVD(n_components=embedding_dim, algorithm='arpack')
    embeddings = svd.fit_transform(matrix)
    print embeddings.shape
    print [sum(svd.explained_variance_ratio_[:i]) for i in range(1, embedding_dim + 1)]
    np.save(output_filepath, embeddings)


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


if __name__ == '__main__':
    #assert len(sys.argv) == 4
    write_spacy_vocab(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    #write_dummy_example(sys.argv[1])
