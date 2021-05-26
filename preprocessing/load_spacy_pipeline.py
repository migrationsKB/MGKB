import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacymoji import Emoji
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from src.preprocessing.preprocessing_sentence import prevent_sentence_boundaries

### modify infixer => do not split up the words with hyphens.
### https://spacy.io/usage/linguistic-features#native-tokenizer-additions###

CONCAT_QUOTES = CONCAT_QUOTES.replace('\'', '')
infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # ✅ Commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
)

infix_re = compile_infix_regex(infixes)


def load_nlp():
    nlp = spacy.load('en_core_web_lg', disable=['tagger','parser','ner'])
    nlp.tokenizer.infix_finditer = infix_re.finditer
    emoji = Emoji(nlp, merge_spans=False)
    nlp.add_pipe(prevent_sentence_boundaries)
    nlp.add_pipe(emoji)
    return nlp


if __name__ == '__main__':
    # text = "Stop kow-towing to the EU. WTO means we get a surplus on tariffs."
    # nlp = load_nlp()
    # doc = nlp(text)
    # print([t.text for t in doc])
    print(CONCAT_QUOTES)

