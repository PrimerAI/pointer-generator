import numpy as np
import spacy
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
    article = 'hi there Michael[PERSON] . this is Japan[GPE] weeeeeeeeeeeee.'
    abstract = '<s> Hi Michael[PERSON] . </s>'
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    with open(out_file, 'wb') as writer:
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

def write_spacy_vocab(vocab_filepath, embeddings_filepath, vocab_size, embedding_dim):
    spacy_vocab = spacy.load('en').vocab
    top_words = []

    for w in spacy_vocab:
        if w.rank > 1.25 * vocab_size:
            continue
        try:
            str(w.lower_)
            top_words.append(w)
        except:
            pass

    top_words.sort(key=lambda w: w.rank)
    top_words = top_words[:vocab_size]

    with open(vocab_filepath, 'w') as f:
        for word in top_words:
            f.write('%s\n' % word.lower_)

    vectors = np.array([w.vector for w in top_words])
    svd = TruncatedSVD(n_components=embedding_dim, algorithm='arpack')
    embeddings = svd.fit_transform(vectors)

    print embeddings.shape
    print [sum(svd.explained_variance_ratio_[:i]) for i in range(1, embedding_dim + 1)]
    np.save(embeddings_filepath, embeddings)


if __name__ == '__main__':
    write_spacy_vocab(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    #write_dummy_example(sys.argv[1])
