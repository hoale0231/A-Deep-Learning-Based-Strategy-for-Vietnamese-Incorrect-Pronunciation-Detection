import Levenshtein as Lev
import torch
from torch import Tensor
from mpvn.vocabs.vocab import Vocabulary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ErrorRate(object):
    """
    Provides inteface of error rate calcuation.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    def __call__(self, targets, y_hats):
        """
        Provides total character distance between targets & y_hats

        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model

        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.vocab.label_to_string(target)
            s2 = self.vocab.label_to_string(y_hat)

            dist, length = self.metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist/total_length

    def metric(self, *args, **kwargs):
        raise NotImplementedError


class WordErrorRate(ErrorRate):
    """ Provides word error rate calcuation. """

    def __init__(self, vocab) -> None:
        super(WordErrorRate, self).__init__(vocab)

    def metric(self, s1: str, s2: str):
        """
        Computes the Unit Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        unit2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts strings)
        w1 = [chr(unit2char[w]) for w in s1.split()]
        w2 = [chr(unit2char[w]) for w in s2.split()]
        dist = Lev.distance(''.join(w1), ''.join(w2))
        length = len(s1.split())
        return dist, length


class CharacterErrorRate(ErrorRate):
    """
    Computes the Character Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to characters.
    """
    def __init__(self, vocab):
        super(CharacterErrorRate, self).__init__(vocab)

    def metric(self, s1: str, s2: str):
        """
        Computes the Character Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to characters.

        Args:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1 = s1.replace(' ', '')
        s2 = s2.replace(' ', '')

        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        dist = Lev.distance(s2, s1)
        length = len(s1.replace(' ', ''))

        return dist, length


def accuracy(y: Tensor, y_hat: Tensor, length: Tensor) -> float:
    length = length.to(torch.long)
    y = torch.concat([y_[:l_] for y_, l_ in zip(y, length)]).cpu()
    y_hat = torch.concat([y_[:l_] for y_, l_ in zip(y_hat, length)]).cpu()
    return accuracy_score(y, y_hat)

def f1(y: Tensor, y_hat: Tensor, length: Tensor, pos_label: int = 1) -> float:
    length = length.to(torch.long)
    y = torch.concat([y_[:l_] for y_, l_ in zip(y, length)]).cpu()
    y_hat = torch.concat([y_[:l_] for y_, l_ in zip(y_hat, length)]).cpu()
    return f1_score(y, y_hat, pos_label=pos_label)

def precision(y: Tensor, y_hat: Tensor, length: Tensor, pos_label: int = 1) -> float:
    length = length.to(torch.long)
    y = torch.concat([y_[:l_] for y_, l_ in zip(y, length)]).cpu()
    y_hat = torch.concat([y_[:l_] for y_, l_ in zip(y_hat, length)]).cpu()
    return precision_score(y, y_hat, pos_label=pos_label)

def recall(y: Tensor, y_hat: Tensor, length: Tensor, pos_label: int = 1) -> float:
    length = length.to(torch.long)
    y = torch.concat([y_[:l_] for y_, l_ in zip(y, length)]).cpu()
    y_hat = torch.concat([y_[:l_] for y_, l_ in zip(y_hat, length)]).cpu()
    return recall_score(y, y_hat, pos_label=pos_label)
