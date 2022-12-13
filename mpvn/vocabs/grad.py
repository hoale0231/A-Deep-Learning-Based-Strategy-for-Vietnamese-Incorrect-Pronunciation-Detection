from mpvn.vocabs.vocab import Vocabulary
from torch import Tensor

class GradVocabulary(Vocabulary):
    """
    Converts label to string for grad dataset.

    Args:
        model_path (str): path of sentencepiece model
    """
    def __init__(self, token_path: str):
        super(GradVocabulary, self).__init__()
        with open(token_path) as fr:
            self.vocab = fr.read().splitlines()
        self.vocab = ['_', ' ', '<s>', '<e>'] + self.vocab
        self.pad_id = 0
        self.blank_id = 0
        self.space_id = 1
        self.sos_id = 2
        self.eos_id = 3
        self.vocab_size = len(self.vocab)
        self.phone_map = dict()
        self.index_map = dict()
        for idx, token in enumerate(self.vocab):
            self.phone_map[token] = idx
            self.index_map[idx] = token            

    def label_to_string(self, labels):
        """ Use a character map and convert integer labels to an phone sequence """
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        return ' '.join([self.index_map[label] for label in labels])
    
    def string_to_label(self, text):
        """ Use a phone map and convert phone sequence to an integer sequence """
        if isinstance(text, str):
            text = text.replace(' ', '- -').split('-')
            # text = list(text)
        if not isinstance(text, list):
            raise Exception("text much be str or list")
        return [self.phone_map[phone] for phone in text if phone != '']
    
class WordVocabulary(Vocabulary):
    """
    Converts label to string for grad dataset.

    Args:
        model_path (str): path of sentencepiece model
    """
    def __init__(self, token_path: str):
        super(WordVocabulary, self).__init__()
        with open(token_path) as fr:
            self.vocab = fr.read().splitlines()
        self.vocab = ['_'] + self.vocab
        self.pad_id = 0
        self.vocab_size = len(self.vocab)
        self.word_map = dict()
        self.index_map = dict()
        for idx, token in enumerate(self.vocab):
            self.word_map[token] = idx
            self.index_map[idx] = token            

    def label_to_string(self, labels):
        """ Use a character map and convert integer labels to an word sequence """
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        return ' '.join([self.index_map[label] for label in labels])
    
    def string_to_label(self, text):
        """ Use a word map and convert word sequence to an integer sequence """
        if isinstance(text, str):
            text = text.split()
            # text = list(text)
        if not isinstance(text, list):
            raise Exception("text much be str or list")
        return [self.word_map[word] for word in text]
