"""
Interface to usim[1] data

The Genia tagger is used to lemmatize the text, and this module
can be used without it (trying to access the tags via the pos_tags
attribute of a Sentence object will raise a ValueError). If you
wish to use the genia tagger, you must download the accompanying
genia.py module, as well as configure the paths to the Genia
executable (GENIA_EXE) and the Genia data(GENIA_DATA).

The Collection class represents the entire usim1 collection, and
can must be initialized with paths to where the sentences and context
data are kept. A convenience method Collection.default is provided
to make this a little easier: set COLLECTION_PATH and CONTEXT_PATH
at the top of the code, and call Collection.default to get an instance
of the class. COLLECTION_PATH should point at the decompressed version
of the Usim1 annotations, which can be obtained from [1]. The context
data comes from SemEval2007 English Lexical Substitution task, and
can be obtained by contacting Diana McCarthy <diana@dianamccarthy.co.uk>.

Marco Lui <mhlui@unimelb.edu.au>, July 2011
Last Updated: November 2012

[1] http://www.katrinerk.com/graded-sense-and-usage-annotation
"""


import csv, os
import re

import lxml.etree as etree
from collections import defaultdict
from itertools import takewhile


GENIA_EXE = '/lt/work/mlui/bin/geniatagger'
GENIA_DATA = '/lt/work/mlui/envs/ds-usim/packages/geniatagger'

COLLECTION_PATH =  '/lt/work/mlui/envs/ds-usim/data/usim1/GradedMeaningAnnotation'
CONTEXT_PATH = '/lt/work/mlui/envs/ds-usim/data/context'

try:
  import genia
  tagger = genia.GeniaTagger(GENIA_EXE, GENIA_DATA)
  tag_cache = {}
except ImportError:
  tagger = None
except OSError, e:
  import errno
  if e.errno == errno.ENOENT:
    tagger = None
  else:
    raise e

class Collection(object):
  """
  Represents the entire usim1 collection.
  """
  def __init__(self, sentence_path, ratings_path, context_path):
    """
    @param sentence_path path to the sentence-level data
    @param ratings_path path to the ratings data
    """
    self.sentence_path = sentence_path 
    self.ratings_path = ratings_path
    self.context_path = context_path
    self.sentences = {}
    self.lemmas = {}
    with open(os.path.join(context_path, 'context_mapping')) as f:
      reader = csv.reader(f, delimiter='\t')
      index = dict(reader)
    self.context_index = index

  @classmethod
  def default(cls, coll=COLLECTION_PATH, context=CONTEXT_PATH):
    sentences_path = os.path.join(coll, 'Data', 'LexicalSubstitutions', 'sentences.xml')
    ratings_path = os.path.join(coll, 'Markup', 'UsageSimilarity', 'usim.ratings') 
    c = cls(sentences_path, ratings_path, context)
    c.init()
    return c

  def init(self):
    # Process the raw sentences
    with open(self.sentence_path) as file:
      tree = etree.fromstring(file.read())

    for item in tree:
      lemma = item.get('item')
      for instance in item:
        id = instance.get('id')
        context = instance.find('context')
        head = context.text
        if isinstance(head, unicode):
          head = head.encode('ascii','replace')
        tail = context[0].tail
        if isinstance(tail, unicode):
          tail = tail.encode('ascii', 'replace')
        
        lemma_id = '%s;%s' % (lemma, id)
        context_path = os.path.join(self.context_path, self.context_index[lemma_id])

        sentence = Sentence(id, head, lemma, tail, context_path)
        self.sentences[id] = sentence
        self.lemmas.setdefault(lemma, Lemma(self, lemma)).add_sentence(id, sentence)

    # Process the ratings data
    with open(self.ratings_path) as file:
      reader = csv.DictReader(file)

      for row in reader:
        lemma = row['lemma']
        if row['user_id'] == 'avg': continue
        self.lemmas[lemma].add_judgment(row['lexsub_id1'], row['lexsub_id2'], row['user_id'], row['judgment'])
    
class Lemma(object):
  pos_map = {'n':'noun', 'v':'verb', 'a':'adjective', 'r':'adverb'}

  def __init__(self, collection, name):
    self.lemma = name
    self.collection = collection
    self.sentences = {}
    self.spairs = {}

  def __eq__(self, other):
    return self.lemma == other.lemma

  def __hash__(self):
    return hash(self.name)

  @property
  def headword(self):
    return self.lemma.split('.')[0]

  @property
  def POS(self):
    label = self.lemma.split('.')[1]
    return self.pos_map.get(label, 'UNKNOWN')
    
  def add_sentence(self, id, sentence):
    self.sentences[id] = sentence

  def add_judgment(self, id1, id2, user_id, value):
    key = id1, id2
    if key not in self.spairs:
      self.spairs[key] = SPair(self.collection, id1, id2, self.lemma)
    self.spairs[key].add_judgment(user_id, value)
    

class Sentence(object):
  def __init__(self, id, head, lemma, tail, context_path):
    self.id = id
    self.head = head
    self.lemma = lemma
    self.tail = tail
    self.context = Context(context_path, self)

  def __eq__(self, other):
    return self.id == other.id

  def __repr__(self):
    return "<Sentence %s (%s)>" % (self.id, self.lemma)

  def __str__(self):
    return self.head + self.headword + self.tail

  @property
  def headword(self):
    return self.lemma.split('.')[0]

  @property
  def pos_tags(self):
    if tagger is None:
      raise ValueError("POS tagging not available as Genia tagger was not detected")
    if self.id not in tag_cache:
      tag_cache[self.id] = tagger.process(str(self))
    return tag_cache[self.id]

  @property
  def text(self):
    return str(self)

RE_ANN = re.compile(r'<id \d+;\d+;\d+ (?P<lemma>\w+)/id>')
def deannotate(line):
  """
  Strip annotations off a line.
  They look line this: "worked <id 422;108;5 hard/id> and"
  Which needs to become this: "worked hard and"
  """
  return RE_ANN.sub('\g<lemma>',line)

class Context(object):
  def __init__(self, path, parent):
    self.path = path
    self.parent = parent
    self.init()

  @property
  def raw(self):
    with open(self.path) as f:
      text = f.read()
    return text

  def init(self):
    raw_iter = iter(self.raw.splitlines())
    lemmas = dict((t[0], t[1:]) for t in map(str.split, takewhile(lambda x:not x.startswith("<text"), raw_iter)))
    self.lines = list(raw_iter)
    parent_id = self.parent.lemma + ';' + self.parent.id
    loc_raw = lemmas[parent_id]
    self.location = int(loc_raw[1]) - 1, int(loc_raw[2]) - 1
        
  @property
  def text(self):
    return '\n'.join(self.lines)

  def around(self, before=1, after=1):
    lineno = self.location[0]
    lines = self.lines[max(0,lineno-before):lineno+after+1]
    return '\n'.join(map(deannotate,lines))

      

class SPair(object):
  def __init__(self, collection, id1, id2, lemma=None):
    self.collection = collection
    s = self.collection.sentences
    if s[id1].lemma != s[id2].lemma:
      raise ValueError, "different lemmas"
    self.id1 = id1
    self.id2 = id2
    if lemma is not None:
      if self.lemma != lemma:
        raise ValueError, "reference lemma does not match"
    self.judgments = {}

  def __repr__(self):
    return "<SPair with %d judgements>" % len(self.judgments)

  @property
  def lemma(self):
    return self.collection.sentences[self.id1].lemma

  @property
  def avg(self):
    return sum(self.judgments.values()) / float(len(self.judgments.values()))

  @property
  def s1(self):
    return self.collection.sentences[self.id1]

  @property
  def s2(self):
    return self.collection.sentences[self.id2]

  def add_judgment(self, user_id, value):
    self.judgments[user_id] = int(value)

def output_annotator_avg():  
  import csv
  c = Collection.default()
  with open('usim1-annotator-avg', 'w') as f:
    writer = csv.writer(f)
    for lemma in c.lemmas.values():
      for spair in lemma.spairs.values():
        writer.writerow( [ spair.id1, spair.id2, spair.avg ] ) 

if __name__ == "__main__":
  c = Collection.default()
  for s in c.sentences.values():
    print s.id, s.context.path


