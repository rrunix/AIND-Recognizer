import math
import statistics
import warnings
import itertools
import operator

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

NEG_INF = float("-inf")


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stdout)
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def get_component_range(self):
        return range(self.min_n_components, min(len(self.X), self.max_n_components + 1))


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score(self, hmm_model):
        try:
            n = len(self.lengths)
            logL = hmm_model.score(self.X, self.lengths)
            logN = np.log(len(self.X))
            p = n * n + 2 * n * len(self.X[0]) - 1
            return -2 * logL + p * logN
        except ValueError:
            return float("-inf")

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Create the models, compute the score using BIC criteria and then get the one with lowest score.
        # (less is better)
        return min([self.base_model(n_components) for n_components in self.get_component_range()], key=self.score)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def score(self, hmm):
        try:
            return hmm.score(self.X, self.lengths)
        except ValueError:
            return float("-inf")

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Create the hmms
        hmms = [self.base_model(n_components) for n_components in self.get_component_range()]

        # Compute the score
        log_scores = np.array([self.score(hmm) for hmm in hmms])

        # Calculate the term SUM(log(P(X(all but i)).
        scores_sum = log_scores.sum()
        log_scores_sum = np.array(list(map(lambda x: scores_sum - x, log_scores)))

        dic_score = log_scores - 1 / log_scores_sum.mean()

        return hmms[np.argmax(dic_score)]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    @staticmethod
    def score(hmm: GaussianHMM, x, lengths):
        return hmm.score(x, lengths)

    def compute_cv(self, hmm: GaussianHMM):
        try:
            n_splits = min(3, len(self.lengths))
            sequences = KFold(n_splits=n_splits).split(self.sequences)

            # Calculate the score for all the splits
            scores = [self.score(self.base_model(hmm.n_components), *combine_sequences(cv_train_idx, self.sequences))
                      for cv_train_idx, _ in sequences]

            if scores:
                # Compute mean
                return sum(scores) / len(scores)
        except:
            pass

        return float("-inf")

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        return max(
            [self.base_model(n_components) for n_components in self.get_component_range()], key=self.compute_cv)
