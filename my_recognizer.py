import warnings
from asl_data import SinglesData


def score(model, x, lengths):
    """
    Compute the score of `x` being represented by `model`
    
    :param model: A model
    :param x: The features
    :param lengths: Lengths of the individual sequences in `x`
    :return: The score or -inf if there is any problem when computing the score
    """
    try:
        return model.score(x, lengths)
    except:
        return float("-inf")


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for x, lengths in test_set.get_all_Xlengths().values():

        # For each word, we compute the probability that this word is actually x.
        word_probabilities = {word : score(model, x, lengths) for word, model in models.items()}

        # Get the word that have the higher chances of being x.
        guess = max(word_probabilities.items(), key=lambda x: x[1])[0]

        probabilities.append(word_probabilities)
        guesses.append(guess)

    return probabilities, guesses