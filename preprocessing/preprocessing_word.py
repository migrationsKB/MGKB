import regex
from nltk.corpus import words
from spellchecker import SpellChecker


def remove_repeated_characters(word, words):
    """
    remove repeated characters from a word, and check if the word exists in wordnet.
    cannot check on hashtags.!
    :param word:
    :return:
    """
    pattern = regex.compile(r"(\w*)(\w)\2(\w*)")
    substitution_pattern = r"\1\2\3"
    while True:
        if word in words:
            return word
        new_word = pattern.sub(substitution_pattern, word)
        if new_word != word:
            word = new_word
            continue
        else:
            return new_word


