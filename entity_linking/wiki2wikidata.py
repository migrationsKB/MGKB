from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from utils.utils import chunks, chunks_dictionary
import time
import json
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np

def get_query(wiki_labels):
    # "Ministry of Defence (Russia)"@en
    # "Wikidata"@en
    QUERY_ = "\n".join(['\"' + label + '\"@en' for label in wiki_labels if label is not np.nan])
    print(QUERY_)
    QUERY_BEF = """
    SELECT ?lemma ?item ?itemLabel ?itemDescription WHERE {
      VALUES ?lemma {

      """

    QUERY_AFTER = """
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
      ?sitelink schema:about ?item;
        schema:isPartOf <https://en.wikipedia.org/>;
        schema:name ?lemma.
    }
    """
    return QUERY_BEF + QUERY_ + QUERY_AFTER


WIKI_LABELS = ["NATO summit", "Secretary General of NATO"]


def run_query(id_):
    df = pd.read_csv("data/extracted/entities_wikipedia.csv", index_col=0)
    # df['entity'] = df['entity'].str.replace('\'s', '').replace('.', '').replace('-', ' ').replace('\"', '')
    df['entity'] = df['entity'].str.replace('\"', '\'')
    wikipedia_ids = df.index.tolist()
    wiki_labels = df["entity"].tolist()
    entities_dict = dict(zip(wikipedia_ids, wiki_labels))
    dl = list(chunks_dictionary(entities_dict, 100))
    item = dl[id_]
    item_labels = list(item.values())
    item_ids = list(item.keys())
    print(item_labels)
    entities_dict = dict()

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    QUERY = get_query(item_labels)
    print(QUERY)

    sparql.setQuery(QUERY)
    sparql.setReturnFormat(JSON)
    q = sparql.query()
    print(q)
    results = q.convert()

    # raise HTTPError(req.full_url, code, msg, hdrs, fp)
    # urllib.error.HTTPError: HTTP Error 429: Too Many Requests
    # if response.status_code == 429:
    #     time.sleep(int(response.headers["Retry-After"]))

    for wiki_id, result in zip(item_ids, results["results"]["bindings"]):
        label = result["lemma"]["value"]
        uri = result["item"]["value"]
        if "itemDescription" in result:
            item_description = result["itemDescription"]["value"]
            entities_dict[wiki_id] = {"wiki_id": label, "wikidata_uri": uri, "wikidata_description": item_description}
        else:
            entities_dict[wiki_id] = {"wiki_id": label, "wikidata_uri": uri, "wikidata_description": ""}

    with open(f"data/extracted/wikidata/event_wiki2data_entities_{idx}.json", "w") as f:
        json.dump(entities_dict, f)
    time.sleep(120)
    id_ += 1
    return id_


def query_loop():
    idx = 55
    while True:
        try:
            idx = run_query(idx)
            print(f"idx ---> {idx}")

        except HTTPError:
            time.sleep(60)
            idx = run_query(idx)
            print(f"idx ---> {idx}")


idx = 446
run_query(idx)
print(f"idx ---> {idx}")
