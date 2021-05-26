import os
import yaml


def load_keywords_for_lang(input_dir, lang):
    keywords_dict = yaml.load(open(os.path.join(input_dir, 'src', 'config', 'lang_keywords.yaml')), yaml.FullLoader)
    return keywords_dict[lang]