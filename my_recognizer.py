import warnings
from asl_data import SinglesData


def score(model, x, lengths):
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

    for idx, (x, lengths) in enumerate(test_set.get_all_Xlengths().values()):
        word_probabilities = {word : score(model, x, lengths) for word, model in models.items()}

        guess = max(word_probabilities.items(), key=lambda x: x[1])[0]

        probabilities.append(word_probabilities)
        guesses.append(guess)

    return probabilities, guesses

