import hashlib
import multiprocessing
import os
import random
import re
import struct
import sys
import time
from collections import Counter
from tensorflow.core.example import example_pb2

from data import ENTITY_TOKENS, POS_TOKENS
from io_processing import process_article
from primer_core.nlp.get_spacy import get_spacy
from pygov.analytic_pipeline.document_pipeline import SingleDocument

dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
# acceptable ways to end a sentence
END_TOKENS = [
    '.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"
]

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506
num_expected_new_cables = 101476

CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
VOCAB_SIZE = 50000

assert all(token[0] == '[' and token[-1] == ']' for token in ENTITY_TOKENS + POS_TOKENS)
ENTITY_TAGS = tuple(token[1: -1] for token in ENTITY_TOKENS)
POS_TAGS = tuple(token[1: -1] for token in POS_TOKENS)


def chunk_file(finished_files_dir, chunks_dir, set_name):
    in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        # new chunk
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(finished_files_dir):
    chunks_dir = os.path.join(finished_files_dir, 'chunked')
    if not os.path.exists(chunks_dir):
        os.mkdir(chunks_dir)

    for set_name in ['train', 'val', 'test']:
        print "Splitting %s data into chunks..." % set_name
        chunk_file(finished_files_dir, chunks_dir, set_name)
    print "Saved chunked data in %s" % chunks_dir


def tokenize_stories(stories_dir, tokenized_stories_dir, is_cable):
    """
    Maps a whole directory of .story files to a tokenized version using spacy.
    """
    print "Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir)
    tasks = multiprocessing.JoinableQueue()
    n_workers = multiprocessing.cpu_count()
    print 'Creating %d workers' % n_workers

    for i in range(n_workers):
        worker = ArticlePreprocesser(tasks, stories_dir, tokenized_stories_dir, is_cable)
        worker.start()

    for story in os.listdir(stories_dir):
        tasks.put(story)
    for i in range(n_workers):
        tasks.put(None)

    tasks.join()

    # Check that the tokenized stories directory contains the same number of files as the
    # original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same "
            "number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig
            )
        )
    print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


class ArticlePreprocesser(multiprocessing.Process):

    def __init__(self, task_queue, input_dir, output_dir, is_cable):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.is_cable = is_cable

    def run(self):
        while True:
            filename = self.task_queue.get()
            if filename is None:
                self.task_queue.task_done()
                break

            input_filename = os.path.join(self.input_dir, filename)
            output_filename = os.path.join(self.output_dir, filename)
            if os.path.isfile(output_filename):
                self.task_queue.task_done()
                continue

            process_task(input_filename, output_filename, self.is_cable)
            self.task_queue.task_done()


def process_task(input_filename, output_filename, is_cable):
    article, abstract = get_art_abs(input_filename, add_periods=True, is_cable=is_cable)
    article = unicode(article, 'utf-8').replace(u'\xa0', ' ')
    abstract = unicode(abstract, 'utf-8').replace(u'\xa0', ' ')

    doc = SingleDocument(0, raw={'body': article})
    clean_article = doc.text()
    full_text = u'%s %s' % (clean_article, abstract)
    spacy_text = get_spacy()(full_text)

    text_tokens, text_token_indices, _ = process_article(spacy_text, print_edge_cases=True)
    article_tokens = [
        text for text, idx in zip(text_tokens, text_token_indices) if idx < len(clean_article)
    ]
    abstract_tokens = [
        text for text, idx in zip(text_tokens, text_token_indices) if idx >= len(clean_article)
    ]

    with open(output_filename, 'w') as f:
        f.write(' '.join(article_tokens).encode('utf-8'))
        f.write('\n\n')
        f.write('@highlight\n')
        f.write(' '.join(abstract_tokens).encode('utf-8'))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """
    Returns a heximal formated SHA1 hash of the input string.
    """
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """
    Adds a period to a line that is missing a period.
    """
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."


def get_art_abs(story_file, add_periods, is_cable):
    if is_cable:
        # ignore add_periods
        return get_art_abs_cable(story_file)
    else:
        return get_art_abs_canonical(story_file, add_periods)


def get_art_abs_canonical(story_file, add_periods):
    lines = read_text_file(story_file)

    if add_periods:
        # Put periods on the ends of lines that are missing them (this is a problem in the dataset
        # because many image captions don't end in periods; consequently they end up in the body of
        # the article as run-on sentences)
        lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article / highlights into a single string
    article = ' '.join(article_lines)
    highlights = ' '.join(highlights)

    return article, highlights


def get_art_abs_cable(story_file):
    lines = read_text_file(story_file)
    assert len(lines) == 1

    summary_match = re.search(
        r'(summary(\s*and\s*comment)?[:.]?\s*)(.*)(end\s*summary(\s*and\s*comment)?\.?\s*)',
        lines[0],
        flags=re.IGNORECASE | re.S,
    )
    assert summary_match is not None

    abstract = summary_match.groups()[2]
    article = lines[0][summary_match.end():].rstrip()
    return ' '.join(article.split()), ' '.join(abstract.split())


def write_to_bin(tokenized_story_dirs, out_dir):
    """
    Joins the .story files into training, validation, and test files.
    """
    all_tokenized_story_paths = []
    for dir in tokenized_story_dirs:
        all_tokenized_story_paths.extend([
            os.path.join(dir, filename) for filename in os.listdir(dir)
        ])

    n_total_paths = len(all_tokenized_story_paths)
    assert n_total_paths == num_expected_cnn_stories + num_expected_dm_stories + num_expected_new_cables
    random.shuffle(all_tokenized_story_paths)

    train_paths = all_tokenized_story_paths[: int(.85 * n_total_paths)]
    validation_paths = all_tokenized_story_paths[int(.85 * n_total_paths): int(.95 * n_total_paths)]
    test_paths = all_tokenized_story_paths[int(.95 * n_total_paths):]

    for input_paths, output_filename in (
        (train_paths, 'train.bin'),
        (validation_paths, 'val.bin'),
        (test_paths, 'test.bin'),
    ):
        out_file = open(os.path.join(out_dir, output_filename), 'wb')
        vocab_counter = Counter()

        for idx, story_path in enumerate(input_paths):
            if idx % 1000 == 0:
                print "Writing story %i of %i; %.2f percent done" % (
                    idx, len(input_paths), 100. * idx / len(input_paths)
                )

            # Get the strings to write to .bin file
            # At this point all articles have been processed so is_cable is False.
            article, abstract = get_art_abs(story_path, add_periods=False, is_cable=False)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            out_file.write(struct.pack('q', str_len))
            out_file.write(struct.pack('%ds' % str_len, tf_example_str))

            # Update vocab counts
            if 'train' in output_filename:
                words = []

                for token in article.split() + abstract.split():
                    bracket_index = token.find('[')
                    curly_bracket_index = token.find('{')
                    assert (bracket_index > 0) ^ (curly_bracket_index > 0)

                    if curly_bracket_index > 0:
                        continue
                    elif token[bracket_index:] in ENTITY_TOKENS:
                        continue

                    word = token[: max(bracket_index, curly_bracket_index)]
                    assert word
                    words.append(word)

                vocab_counter.update(words)

        out_file.close()

        print "Finished writing file %s\n" % output_filename

        if 'train' in output_filename:
            print 'Writing vocab file'
            with open(os.path.join(out_dir, 'vocab'), 'w') as out:
                for word, count in vocab_counter.most_common(VOCAB_SIZE):
                    out.write('%s %d\n' % (word, count))


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (
                stories_dir, num_stories, num_expected
            )
        )


def main():
    if len(sys.argv) != 3:
        print "USAGE: python make_datafiles.py <raw_stories_dir> <output_dir>"
        sys.exit()

    # Define input / output directories
    raw_stories_dir = sys.argv[1]
    output_dir = sys.argv[2]

    cnn_stories_dir = os.path.join(raw_stories_dir, 'cnn')
    dm_stories_dir = os.path.join(raw_stories_dir, 'dailymail')
    cables_stories_dir = os.path.join(raw_stories_dir, 'cables')
    cnn_tokenized_stories_dir = os.path.join(output_dir, 'cnn_stories_tokenized')
    dm_tokenized_stories_dir = os.path.join(output_dir, 'dm_stories_tokenized')
    cables_tokenized_stories_dir = os.path.join(output_dir, 'cables_stories_tokenized')
    finished_files_dir = os.path.join(output_dir, 'finished_files')

    # Make some output directories
    for dirname in (
        output_dir, cnn_tokenized_stories_dir, dm_tokenized_stories_dir,
        cables_tokenized_stories_dir, finished_files_dir
    ):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Check the stories directories contain the correct number of .story files
    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    check_num_stories(dm_stories_dir, num_expected_dm_stories)
    check_num_stories(cables_stories_dir, num_expected_new_cables)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir, is_cable=False)
    tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir, is_cable=False)
    tokenize_stories(cables_stories_dir, cables_tokenized_stories_dir, is_cable=True)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(
        (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, cables_tokenized_stories_dir),
        finished_files_dir,
    )

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks,
    # each containing e.g. 1000 examples, and saves them in finished_files/chunks.
    chunk_all(finished_files_dir)


if __name__ == '__main__':
    main()
