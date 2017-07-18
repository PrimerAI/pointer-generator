import json
import numpy as np
import os
import spacy
import string
import struct
import sys
from sklearn.decomposition.truncated_svd import TruncatedSVD
from tensorflow.core.example import example_pb2

from data import N_FREE_TOKENS, Vocab
from generate import generate_summary
from make_datafiles import get_art_abs
from pygov.analytic_pipeline.common.summary import compute_summaries
from pygov.analytic_pipeline.document_pipeline import SingleDocument


######################################################
# Handling vocab
######################################################

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

        if i >= N_FREE_TOKENS and unicode(word) not in spacy_vocab:
            continue

        if i >= N_FREE_TOKENS:
            final_vocab.append(word)

        matrix[new_i] = spacy_vocab[unicode(word)].vector
        new_i += 1

    print 'Last word added:', final_vocab[-1]
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

# NOTE: Don't use this anymore - is not consistent with how Vocab loads words
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

######################################################
# Testing
######################################################

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

######################################################
# Comparing results
######################################################

RAW_DATA_DIR = '/Users/michaelwu/dev/cnn-dailymail/raw_data/'
RAW_ARTICLE_DIRS = (os.path.join(RAW_DATA_DIR, dir) for dir in ('cnn', 'dailymail'))

RESULTS_DIR = '/Users/michaelwu/dev/text-summarization/results/'
RESULTS_ARTICLE_DIR = os.path.join(RESULTS_DIR, 'articles')
RESULTS_ABSTRACT_DIR = os.path.join(RESULTS_DIR, 'abstract')

# tuple of output name, dir, filename
SUMMARY_OUTPUT_LOCATIONS = (
    ('Reference', RESULTS_ABSTRACT_DIR, 'abstract_%d.txt'),
    #('Normal', os.path.join(RESULTS_DIR, 'decoded_normal'), '%06d_decoded.txt'),
    #('Coverage', os.path.join(RESULTS_DIR, 'decoded_coverage'), '%06d_decoded.txt'),
    #('Coverage v4', os.path.join(RESULTS_DIR, 'decoded_coverage_4'), '%06d_decoded.txt'),
    #('Restrictive', os.path.join(RESULTS_DIR, 'decoded_restr'), '%06d_decoded.txt'),
    #('Corrective', os.path.join(RESULTS_DIR, 'decoded_corrective'), '%06d_decoded.txt'),
    #('Abisee', os.path.join(abisee_result_dir, 'pointer-gen-cov'), '%s_decoded.txt'),
)

N_ARTICLES = 100

SEARCH_TERMS = {
    'tesla',
    'crispr',
    'bitcoin',
    'zika',
    'wall street',
    'tim cook',
    'merkel',
    'intelligence',
    'amazon',
    'microsoft',
    'FBI',
    'vladimir putin',
    'zuckerburg',
}

def find_articles():
    if os.path.exists(RESULTS_DIR):
        raise Exception

    os.mkdir(RESULTS_DIR)
    os.mkdir(RESULTS_ARTICLE_DIR)
    os.mkdir(RESULTS_ABSTRACT_DIR)

    articles = []

    for article_dir in RAW_ARTICLE_DIRS:
        for filename in os.listdir(article_dir):
            article_path = os.path.join(article_dir, filename)
            article, abstract = get_art_abs(article_path, add_periods=True)

            article_words = article.lower().split()
            full_article_words = set(
                article_words + [
                    ' '.join([article_words[i], article_words[i + 1]])
                    for i in range(len(article_words) - 1)
                ]
            )
            search_count = sum(1 for term in SEARCH_TERMS if term in full_article_words)
            if search_count:
                articles.append((article, abstract, search_count))

    articles.sort(key=lambda info: info[2], reverse=True)
    for i, (article, abstract, search_count) in enumerate(articles[:N_ARTICLES]):
        article_path = os.path.join(RESULTS_ARTICLE_DIR, 'article_%d.txt' % i)
        abstract_path = os.path.join(RESULTS_ABSTRACT_DIR, 'abstract_%d.txt' % i)
        print '#####################'
        print i
        print abstract

        with open(article_path, 'w') as f:
            f.write(article)
        with open(abstract_path, 'w') as f:
            f.write(abstract)

def get_lexrank_summary(doc):
    summaries = compute_summaries(
        [0],
        {0: doc.text()},
        {0: [{'start': span[0], 'end': span[1]} for span in doc.sentence_spans()]},
        {},
    )
    return summaries[0]['summary']

def write_results(out_file):
    out = open(out_file, 'w')
    out.write(
        '\t'.join(
            [name for name, dir, filename in SUMMARY_OUTPUT_LOCATIONS] +
            ['Lexrank', 'Seq-to-seq ready']
        ) + '\n'
    )

    for filename in os.listdir(RESULTS_ARTICLE_DIR):
        article_id = int(filename.split('.')[0].split('_')[1])

        # Read article
        with open(os.path.join(RESULTS_ARTICLE_DIR, filename)) as f:
            article_text = unicode(f.read(), 'utf-8').replace(u'\xa0', ' ')

        # Read pregenerated seq-to-seq summaries
        summaries = []
        for name, summary_dir, summary_filename in SUMMARY_OUTPUT_LOCATIONS:
            with open(os.path.join(summary_dir, summary_filename % article_id)) as f:
                summaries.append(f.read())

        doc = SingleDocument(0, raw={'body': article_text})

        # Generate lexrank summary on the fly
        summaries.append(get_lexrank_summary(doc))

        # Generate seq-to-seq summary on the fly
        spacy_article = doc.spacy_text()
        summaries.append(generate_summary(spacy_article))

        # Print all results together
        for i, summ in enumerate(summaries):
            if isinstance(summ, unicode):
                summ = summ.encode('utf-8')
            out.write(summ.replace('\t', ' ').replace('\n', ' '))
            if i == len(summaries) - 1:
                out.write('\n')
            else:
                out.write('\t')

        out.flush()

    out.close()

######################################################
# Generate sample summaries
######################################################

def get_cable_results(data_file, out_file):
    out = open(out_file, 'w')
    out.write('\t'.join(['Cable', 'Lexrank', 'Seq-to-seq']) + '\n')

    with open(data_file) as f:
        cables = json.load(f)

    for cable in cables[:100]:
        doc = SingleDocument(0, raw={'body': cable})
        if len(doc.text()) < 500:
            continue

        lexrank = get_lexrank_summary(doc)
        seq2seq = generate_summary(doc.spacy_text())

        out.write(
            '\t'.join([
                string.encode('utf-8').replace('\t', ' ').replace('\n', ' ')
                for string in [cable, lexrank, seq2seq]
            ]) + '\n'
        )
        out.flush()

    out.close()


def print_results():
    for filename in os.listdir(RESULTS_ARTICLE_DIR):
        with open(os.path.join(RESULTS_ARTICLE_DIR, filename)) as f:
            article_text = unicode(f.read(), 'utf-8').replace(u'\xa0', ' ')

        doc = SingleDocument(0, raw={'body': article_text})
        spacy_article = doc.spacy_text()
        print '#################'
        print generate_summary(spacy_article)


if __name__ == '__main__':
    #compute_reduced_embeddings_original_vocab(
    #    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    #)
    write_results(sys.argv[1])
    #find_articles()
    #generate_input_file(sys.argv[1])
    #get_cable_results(sys.argv[1], sys.argv[2])
    #print_results()
