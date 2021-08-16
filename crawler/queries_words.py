import json
from itertools import chain
from utils.utils import chunks

with open('../data/extracted/1st_round/chunked_hashtags.json') as file:
    hashtags = json.load(file)

keywords = ['immigrant', 'immigrants', 'refugee', 'refugees', 'asylum',
            'migrant', 'migrants', 'internally displaced', 'UNHCR', 'asylee',
            'asylees', 'Asylee', 'resettled', 'resettlements',
            'immigration', 'Ateh', 're settlement', 'resettle', 'resettles',
            'Dadaab', 'statelessness', 'Hagadera', 'Domiz',
            'émigré', 'exile', 'displaced person', 'deserter', 'pariah', 'pariahs',
            "#refugeesnotwelcome",
            "#DeportallMuslims",
            "#banislam",
            "#banmuslims",
            "#destroyislam",
            "#norefugees",
            "#nomuslims",
            "muslim",
            "islam",
            "islamic",
            "immigration",
            "migrant",
            "refugee",
            "asylum"
            ]

hashtags = list(chain.from_iterable(hashtags))

print(len(hashtags))

all_keywords = hashtags + keywords

print(len(all_keywords))
print(all_keywords[:10])

print(len(' OR '.join(all_keywords[:30])))

all_chunks = list(chunks(all_keywords, 30))
print(len(all_chunks))  # 58
with open('../data/extracted/keywords_chunked_all.json', 'w') as file:
    json.dump(all_chunks, file)
