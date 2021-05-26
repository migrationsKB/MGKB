import re

from preprocessing.contractions import contractions_dict

# https://github.com/explosion/spaCy/issues/2627#issuecomment-410513016
def prevent_sentence_boundaries(doc):
    for token in doc:
        if not can_be_sentence_start(token):
            token.is_sent_start = False
    return doc


def can_be_sentence_start(token):
    """
    check if the token can be the start of a sentence.
    :param token:
    :return:
    """
    if token.i == 0:
        return True
    # We're not checking for is_title here to ignore arbitrary titlecased
    # tokens within sentences
    # elif token.is_title:
    #    return True
    elif token.nbor(-1).is_punct:
        return True
    elif token.nbor(-1).is_space:
        return True
    else:
        return False


c_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions_dict[match.group(0)]
    return c_re.sub(replace, text)
