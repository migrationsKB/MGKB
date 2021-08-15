import os

import yaml


class CountryInfo:
    def __init__(self, input_dir, type):
        self.input_dir = input_dir
        # HOST or SOURCE
        self.type = type
        geoInfo = yaml.load(open(os.path.join(self.input_dir, 'src', 'config', 'geoInfo.yaml')), yaml.FullLoader)
        self.country_langs = geoInfo[self.type]

    def get_langs_by_country(self, COUNTRY):
        return self.country_langs[COUNTRY]['LANGS']

    def get_ISO_by_country(self, COUNTRY):
        return self.country_langs[COUNTRY]['ISO2']


if __name__ == '__main__':
    input_dir = os.getcwd()
    country_info = CountryInfo(input_dir, 'SOURCE')
    langs = country_info.get_langs_by_country('MALI')
    print(langs)
    print(country_info.get_ISO_by_country('MALI'))
