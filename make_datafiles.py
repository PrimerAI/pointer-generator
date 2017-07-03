import sys
import os
import hashlib
import multiprocessing
import struct
import time
from collections import defaultdict
from tensorflow.core.example import example_pb2

from data import ENTITY_TOKENS, POS_TOKENS
from primer_core.entities.people import SpacyPeopleResolver
from primer_core.nlp.get_spacy import get_spacy
from pygov.analytic_pipeline.document_pipeline import SingleDocument


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

assert all(token[0] == '[' and token[-1] == ']' for token in ENTITY_TOKENS + POS_TOKENS)
ENTITY_TAGS = tuple(token[1: -1] for token in ENTITY_TOKENS)
POS_TAGS = tuple(token[1: -1] for token in POS_TOKENS)


def chunk_file(finished_files_dir, chunks_dir, set_name):
  in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
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


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """
  Maps a whole directory of .story files to a tokenized version using spacy
  """
  print "Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir)
  tasks = multiprocessing.JoinableQueue()
  n_workers = multiprocessing.cpu_count()
  print 'Creating %d workers' % n_workers

  for i in range(n_workers):
    worker = ArticlePreprocesser(tasks, stories_dir, tokenized_stories_dir)
    worker.start()

  for story in os.listdir(stories_dir):
    tasks.put(story)
  for i in range(n_workers):
    tasks.put(None)

  tasks.join()

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


class ArticlePreprocesser(multiprocessing.Process):

  def __init__(self, task_queue, input_dir, output_dir):
    multiprocessing.Process.__init__(self)
    self.task_queue = task_queue
    self.input_dir = input_dir
    self.output_dir = output_dir


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

      process_task(input_filename, output_filename)
      self.task_queue.task_done()


def process_task(input_filename, output_filename):
  article, abstract = get_art_abs(input_filename)
  article = unicode(article, 'utf-8').replace(u'\xa0', ' ')
  abstract = unicode(abstract, 'utf-8').replace(u'\xa0', ' ')
  article_tokens, abstract_tokens = process_article_abstract(input_filename, article, abstract)

  with open(output_filename, 'w') as f:
    f.write(' '.join(article_tokens).encode('utf-8'))
    f.write('\n\n')
    f.write('@highlight\n')
    f.write(' '.join(abstract_tokens).encode('utf-8'))


def process_article_abstract(story_name, article, abstract):
  doc = SingleDocument(0, raw={'body': article})
  clean_article = doc.clean_text_and_raw_spans()[0]
  full_article = u'%s %s' % (clean_article, abstract)
  spacy_text = get_spacy()(full_article)
  assert len(full_article) == len(spacy_text.text)

  span_to_person_id, people_resolver = resolve_people(spacy_text, full_article)
  article_tokens = []
  abstract_tokens = []

  for token in spacy_text:
    token_text = token.text.strip().lower()
    if not token_text:
      continue

    person_id = find_person_span_and_update(
      full_article, span_to_person_id, token.idx, token.idx + len(token.text)
    )
    if person_id is not None:
      token_text += '{%d}' % person_id
    elif token.ent_type_ in ENTITY_TAGS:
      token_text += '[%s]' % token.ent_type_
    elif token.pos_ in POS_TAGS:
      token_text += '[%s]' % token.pos_

    if token.idx < len(clean_article):
      article_tokens.append(token_text)
    else:
      abstract_tokens.append(token_text)

  if span_to_person_id:
    print '################'
    print "Person mention not fully found:"
    print story_name
    print span_to_person_id

  return article_tokens, abstract_tokens


def resolve_people(spacy_text, text):
  people_resolver = SpacyPeopleResolver(
    {0: [spacy_text]},
    min_num_persons=1,
    min_person_label_ratio=.1,
    min_p_entity=.1,
    min_p_person=.3,
    min_unambiguous_p=.5,
  )
  people_resolver.resolve(min_p=.5)

  person_to_span = defaultdict(list)
  for key, person_id in people_resolver.key_to_person_root_.iteritems():
    span = people_resolver.occurrences_[key][1][0]
    span = strip_span(span, text[span[0]: span[1]])
    person_to_span[person_id].append(span)

  # person id sorted by count and then order of appearance (id 0 is most popular)
  spans_by_person = sorted(
    person_to_span.values(),
    key=lambda spans: 100 * len(spans) - min(spans)[0],
    reverse=True
  )
  span_to_person_id = {span: i for i, spans in enumerate(spans_by_person) for span in spans}

  return span_to_person_id, people_resolver


def find_person_span_and_update(text, span_to_person_id, start, end):
  span = find_span(span_to_person_id, start, end)
  if span is None:
    return None

  person_id = span_to_person_id.pop(span)
  remaining_mention = text[end: span[1]].lstrip()
  if remaining_mention:
    span_to_person_id[(span[1] - len(remaining_mention), span[1])] = person_id

  return person_id

def find_span(span_to_person_id, start, end):
  for (span_start, span_end), person_id in span_to_person_id.iteritems():
    if start >= span_start and end <= span_end:
      return span_start, span_end
  return None


def strip_span(span, text):
  start = span[0] + len(text) - len(text.lstrip())
  end = span[1] - (len(text) - len(text.rstrip()))
  return start, end


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + "."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx, line in enumerate(lines):
    if line == "":
      continue # empty line
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

def write_to_bin(url_file, cnn_tokenized_stories_dir, dm_tokenized_stories_dir, out_file):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print "Making bin file for URLs listed in %s..." % url_file
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  with open(out_file, 'wb') as writer:
    for idx, s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx) * 100.0 / float(num_stories))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
        story_file = os.path.join(cnn_tokenized_stories_dir, s)
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_tokenized_stories_dir, s)
      else:
        print "Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        # Check again if tokenized stories directories contain correct number of files
        print "Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
        check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, s))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

  print "Finished writing file %s\n" % out_file


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


def main():
  if len(sys.argv) != 4:
    print "USAGE: python make_datafiles.py <raw_stories_dir> <train_test_split_dir> <output_dir>"
    sys.exit()

  # Define input / output directories
  raw_stories_dir = sys.argv[1]
  train_test_split_dir = sys.argv[2]
  output_dir = sys.argv[3]

  cnn_stories_dir = os.path.join(raw_stories_dir, 'cnn')
  dm_stories_dir = os.path.join(raw_stories_dir, 'dailymail')
  cnn_tokenized_stories_dir = os.path.join(output_dir, 'cnn_stories_tokenized')
  dm_tokenized_stories_dir = os.path.join(output_dir, 'dm_stories_tokenized')
  finished_files_dir = os.path.join(output_dir, 'finished_files')

  # Make some output directories
  for dirname in (
    output_dir, cnn_tokenized_stories_dir, dm_tokenized_stories_dir, finished_files_dir
  ):
    if not os.path.exists(dirname):
      os.makedirs(dirname)

  # Check the stories directories contain the correct number of .story files
  check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)
  tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  for dataset in ('train', 'val', 'test'):
    write_to_bin(
      os.path.join(train_test_split_dir, "all_%s.txt" % dataset),
      cnn_tokenized_stories_dir,
      dm_tokenized_stories_dir,
      os.path.join(finished_files_dir, "%s.bin" % dataset),
    )

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all(finished_files_dir)


if __name__ == '__main__':
  main()
