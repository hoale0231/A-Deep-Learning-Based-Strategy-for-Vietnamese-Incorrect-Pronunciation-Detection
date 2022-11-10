from mpvn.vocabs.vocab import Vocabulary

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
        self.vocab = ['<pad>', '<s>', '</s>'] + self.vocab
        self.pad_id = 0
        self.blank_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.vocab_size = len(self.vocab)
        self.phone_map = dict()
        self.index_map = dict()
        for idx, token in enumerate(self.vocab):
            self.phone_map[token] = idx
            self.index_map[idx] = token            

    def label_to_string(self, labels):
        """ Use a character map and convert integer labels to an phone sequence """
        return [[self.index_map[i] for i in sample] for sample in labels]
    
    def string_to_label(self, text):
        """ Use a phone map and convert phone sequence to an integer sequence """
        if isinstance(text, str):
            text = text.split()
        if not isinstance(text, list):
            raise "text much be str or list"
        return [self.phone_map[phone] for phone in text]
