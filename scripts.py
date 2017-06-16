import numpy as np
from sklearn.decomposition.truncated_svd import TruncatedSVD
import spacy
import sys
from data import Vocab


def compute_word_embedding_matrix(output_filepath, vocab_filepath, vocab_size, embedding_dim):
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

if __name__ == '__main__':
    compute_word_embedding_matrix(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
