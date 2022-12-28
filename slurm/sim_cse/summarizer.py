# from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from transformers import AutoTokenizer

# union of sklearn and ntlk stopwords from these two imports:
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
stopwords = set([
    'a',
    'about',
    'above',
    'across',
    'after',
    'afterwards',
    'again',
    'against',
    'ain',
    'all',
    'almost',
    'alone',
    'along',
    'already',
    'also',
    'although',
    'always',
    'am',
    'among',
    'amongst',
    'amoungst',
    'amount',
    'an',
    'and',
    'another',
    'any',
    'anyhow',
    'anyone',
    'anything',
    'anyway',
    'anywhere',
    'are',
    'aren',
    'around',
    'as',
    'at',
    'back',
    'be',
    'became',
    'because',
    'become',
    'becomes',
    'becoming',
    'been',
    'before',
    'beforehand',
    'behind',
    'being',
    'below',
    'beside',
    'besides',
    'between',
    'beyond',
    'bill',
    'both',
    'bottom',
    'but',
    'by',
    'call',
    'can',
    'cannot',
    'cant',
    'co',
    'con',
    'could',
    'couldn',
    'couldnt',
    'cry',
    'd',
    'de',
    'describe',
    'detail',
    'did',
    'didn',
    'do',
    'does',
    'doesn',
    'doing',
    'don',
    'done',
    'down',
    'due',
    'during',
    'each',
    'eg',
    'eight',
    'either',
    'eleven',
    'else',
    'elsewhere',
    'empty',
    'enough',
    'etc',
    'even',
    'ever',
    'every',
    'everyone',
    'everything',
    'everywhere',
    'except',
    'few',
    'fifteen',
    'fify',
    'fill',
    'find',
    'fire',
    'first',
    'five',
    'for',
    'former',
    'formerly',
    'forty',
    'found',
    'four',
    'from',
    'front',
    'full',
    'further',
    'get',
    'give',
    'go',
    'had',
    'hadn',
    'has',
    'hasn',
    'hasnt',
    'have',
    'haven',
    'having',
    'he',
    'hence',
    'her',
    'here',
    'hereafter',
    'hereby',
    'herein',
    'hereupon',
    'hers',
    'herself',
    'him',
    'himself',
    'his',
    'how',
    'however',
    'hundred',
    'i',
    'ie',
    'if',
    'in',
    'inc',
    'indeed',
    'interest',
    'into',
    'is',
    'isn',
    'it',
    'its',
    'itself',
    'just',
    'keep',
    'last',
    'latter',
    'latterly',
    'least',
    'less',
    'll',
    'ltd',
    'm',
    'ma',
    'made',
    'many',
    'may',
    'me',
    'meanwhile',
    'might',
    'mightn',
    'mill',
    'mine',
    'more',
    'moreover',
    'most',
    'mostly',
    'move',
    'much',
    'must',
    'mustn',
    'my',
    'myself',
    'name',
    'namely',
    'needn',
    'neither',
    'never',
    'nevertheless',
    'next',
    'nine',
    'no',
    'nobody',
    'none',
    'noone',
    'nor',
    'not',
    'nothing',
    'now',
    'nowhere',
    'o',
    'of',
    'off',
    'often',
    'on',
    'once',
    'one',
    'only',
    'onto',
    'or',
    'other',
    'others',
    'otherwise',
    'our',
    'ours',
    'ourselves',
    'out',
    'over',
    'own',
    'part',
    'per',
    'perhaps',
    'please',
    'put',
    'rather',
    're',
    's',
    'same',
    'see',
    'seem',
    'seemed',
    'seeming',
    'seems',
    'serious',
    'several',
    'shan',
    'she',
    'should',
    'shouldn',
    'show',
    'side',
    'since',
    'sincere',
    'six',
    'sixty',
    'so',
    'some',
    'somehow',
    'someone',
    'something',
    'sometime',
    'sometimes',
    'somewhere',
    'still',
    'such',
    'system',
    't',
    'take',
    'ten',
    'than',
    'that',
    'the',
    'their',
    'theirs',
    'them',
    'themselves',
    'then',
    'thence',
    'there',
    'thereafter',
    'thereby',
    'therefore',
    'therein',
    'thereupon',
    'these',
    'they',
    'thick',
    'thin',
    'third',
    'this',
    'those',
    'though',
    'three',
    'through',
    'throughout',
    'thru',
    'thus',
    'to',
    'together',
    'too',
    'top',
    'toward',
    'towards',
    'twelve',
    'twenty',
    'two',
    'un',
    'under',
    'until',
    'up',
    'upon',
    'us',
    've',
    'very',
    'via',
    'was',
    'wasn',
    'we',
    'well',
    'were',
    'weren',
    'what',
    'whatever',
    'when',
    'whence',
    'whenever',
    'where',
    'whereafter',
    'whereas',
    'whereby',
    'wherein',
    'whereupon',
    'wherever',
    'whether',
    'which',
    'while',
    'whither',
    'who',
    'whoever',
    'whole',
    'whom',
    'whose',
    'why',
    'will',
    'with',
    'within',
    'without',
    'won',
    'would',
    'wouldn',
    'y',
    'yet',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves'
])


class Summarizer:
    """To summarize a data entry pair into length up to the max sequence length.

    Args:
        task_config (Dictionary): the task configuration
        lm (string): the language model (bert, albert, or distilbert)

    Attributes:
        config (Dictionary): the task configuration
        tokenizer (Tokenizer): a tokenizer from the huggingface library
    """
    def __init__(self, entries, lm):
        self.entries = entries
        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.len_cache = {}

        # build the tfidf index
        self._build_index()

    def _build_index(self):
        """Build the idf index.

        Store the index and vocabulary in self.idf and self.vocab.
        """
        vectorizer = TfidfVectorizer().fit(self.entries)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    def _get_len(self, word):
        """Return the sentence_piece length of a token.
        """
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length

    # def _stem(self, text):
    #     porter = PorterStemmer()
    #     return [porter.stem(word) for word in text.split(' ')]

    def transform(self, line, max_len):
        """Summarize one single example.

        Only retain tokens of the highest tf-idf

        Args:
            row (str): a matching example of two data entries and a binary label, separated by tab
            max_len (int, optional): the maximum sequence length to be summarized to

        Returns:
            str: the summarized example
        """
        res = ''
        cnt = Counter()
        entry = line.lower()

        tokens = line.split(' ')
        for token in tokens:
            if token not in stopwords and token in self.vocab:
                cnt[token] += self.idf[self.vocab[token]]

        token_cnt = Counter(tokens)

        subset = Counter()
        for token in set(token_cnt.keys()):
            subset[token] = cnt[token]
        subset = subset.most_common(max_len)

        topk_tokens_copy = set([])
        total_len = 0
        for word, _ in subset:
            bert_len = self._get_len(word)
            if total_len + bert_len > max_len:
                break
            total_len += bert_len
            topk_tokens_copy.add(word)

        for token in tokens:
            if token in topk_tokens_copy:
                res += token + ' '
                topk_tokens_copy.remove(token)

        return res.rstrip()

    def transform_lines(self, lines, max_len=64):
        return [self.transform(line, max_len=max_len) for line in lines]

