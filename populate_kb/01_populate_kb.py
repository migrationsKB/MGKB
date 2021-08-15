import json
from datetime import datetime
from ast import literal_eval

import rdflib
import pandas as pd
import numpy as np
import uuid

from rdflib.namespace import Namespace, RDF
from rdflib import URIRef, Literal

### load the graph from current owl file.
g = rdflib.Graph()
# g.parse('input/migrationsKB_schema.owl', format='application/rdf+xml')
g.parse('input/migrationsKB_schema.ttl', format='ttl')

# define namespace
sioc = Namespace('http://rdfs.org/sioc/ns#')
sioc_t = Namespace('http://rdfs.org/sioc/types#')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
rdf = Namespace('https://www.w3.org/1999/02/22-rdf-syntax-ns#')
nee = Namespace('http://www.ics.forth.gr/isl/oae/core#')
schema = Namespace('http://schema.org/')
onyx = Namespace('http://www.gsi.dit.upm.es/ontologies/onyx/ns#')
wna = Namespace('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#')
dc = Namespace('http://purl.org/dc/elements/1.1/')
fibo_ind_ei_ei = Namespace('https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/')
fibo_fnd_dt_fd = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/DatesAndTimes/FinancialDates/')
fibo_fnd_rel_rel = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/')
fibo_fnd_utl_alx = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Utilities/Analytics/')
fibo_fnd_arr_doc = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Arrangements/Documents/')
fibo_fnd_arr_rep = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Arrangements/Reporting/')
fibo_fnd_arr_asmt = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Arrangements/Assessments/')
prov = Namespace('https://www.w3.org/TR/prov-o/#')
MGKB = "https://migrationsKB.github.io/MGKB#"
mgkb = Namespace(MGKB)

## binding
g.bind('mgkb', mgkb)
g.bind("sioc", sioc)
g.bind("sioc_t", sioc_t)
g.bind('rdf', rdf)
g.bind("rdfs", rdfs)
g.bind("wna", wna)
g.bind("nee", nee)
g.bind("dc", dc)
g.bind("schema", schema)
g.bind("onyx", onyx)
g.bind("fibo-ind-ei-ei", fibo_ind_ei_ei)
g.bind("fibo-fnd-dt-fd", fibo_fnd_dt_fd)
g.bind("fibo-fnd-rel-rel", fibo_fnd_rel_rel)
g.bind("fibo-fnd-utl-alx", fibo_fnd_utl_alx)
g.bind("fibo-fnd-arr-doc", fibo_fnd_arr_doc)

### defined individuals
neutral_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#neutral-emotion')
negative_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#negative-emotion')
positive_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#positive-emotion')

hate_speech = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#hate')
offensive_speech = mgkb.offensive
normal_speech = mgkb.normal

youth_unemployed_population = URIRef(MGKB + 'youth-unemployed-population')
total_unemployed_population = URIRef(MGKB + 'total-unemployed-population')

statista = mgkb.statista
eurostat = mgkb.eurostat
ONS = mgkb.ONS
UKParliament = mgkb.UKParliament

### accessment-activities
assessment_activity_eurostat = URIRef(MGKB + 'assessment-activity-eurostat')
assessment_activity_ons = URIRef(MGKB + 'assessment-activity-ons')
assessment_activity_ukparliament = URIRef(MGKB + 'assessment-activity-uk-parliament')
assessment_activity_statista = URIRef(MGKB + 'assessment-activity-statista')

g.add((assessment_activity_statista, prov.wasAssociatedWith, statista))
g.add((assessment_activity_eurostat, prov.wasAssociatedWith, eurostat))
g.add((assessment_activity_ons, prov.wasAssociatedWith, ONS))
g.add((assessment_activity_ukparliament, prov.wasAssociatedWith, UKParliament))


def define_entity_resources(entities_dict):
    """
    rdfs:Resource (wikipedia urls)
    """
    for idx, ent_dict in entities_dict.items():
        ent_instance = URIRef(ent_dict['url'])
        ent_label = ent_dict['entity']
        ent_description = ent_dict['description']
        g.add((ent_instance, RDF.type, rdfs.Resource))  # individual of rdfs.Resource
        g.add((ent_instance, rdfs.label, Literal(ent_label)))
        g.add((ent_instance, schema.description, Literal(ent_description)))


def define_economic_indicators():
    real_gdp_r = pd.read_csv('../data/eurostat_stats/csv/real_gdp_growth_rate.csv',
                             index_col=0)
    total_unemployment_rate = pd.read_csv('../data/eurostat_stats/csv/unemployment_rate.csv',
                                          index_col=0)
    youth_unemployment_rate = pd.read_csv(
        '../data/eurostat_stats/csv/youth_unemployment_rate.csv', index_col=0)

    rgdpr_dict = real_gdp_r.to_dict()
    total_ur = total_unemployment_rate.to_dict()
    youth_ur = youth_unemployment_rate.to_dict()

    # report date instances.
    rgdpr_report_date_eurostat_instance = URIRef(MGKB + 'rgdpr_eurostat_report_date')  # rgdpr eurostat
    g.add((rgdpr_report_date_eurostat_instance, RDF.type, fibo_fnd_dt_fd.ExplicitDate))
    g.add((rgdpr_report_date_eurostat_instance, fibo_fnd_dt_fd.hasDateValue, Literal('2021-05-12')))

    rgdpr_report_date_GB_2020_instance = URIRef(MGKB + 'rgdpr_gb_2020_report_date')
    g.add((rgdpr_report_date_GB_2020_instance, RDF.type, fibo_fnd_dt_fd.ExplicitDate))
    g.add((rgdpr_report_date_GB_2020_instance, fibo_fnd_dt_fd.hasDateValue, Literal('2021-03-31')))

    ### real gdp growth rate.
    for country, year_dict in rgdpr_dict.items():
        for year, rate in year_dict.items():
            rgdpr_instance = URIRef(MGKB + 'rgdpr_' + country + '_' + str(year))
            g.add((rgdpr_instance, RDF.type, fibo_ind_ei_ei.GrossDomesticProduct))
            g.add((rgdpr_instance, fibo_ind_ei_ei.hasIndicatorValue, Literal(rate)))  # hasIndicatorValue.
            g.add((rgdpr_instance, schema.addressCountry, Literal(country)))  # schema.addressCountry
            g.add((rgdpr_instance, dc.date, Literal(year)))  # dc.date
            if year == 2020 and country == 'GB':
                g.add((rgdpr_instance, prov.wasGeneratedBy, statista))
                g.add((rgdpr_instance, fibo_fnd_arr_rep.hasReportDate,
                       rgdpr_report_date_GB_2020_instance))  # hasReportDate
            else:
                g.add((rgdpr_instance, prov.wasGeneratedBy, eurostat))
                g.add((rgdpr_instance, fibo_fnd_arr_rep.hasReportDate, rgdpr_report_date_eurostat_instance))

    ### total unemployment rate report date instances
    tur_report_date_eurostat_instance = URIRef(MGKB + 'tur_eurostat_report_date')  # total unemployment eurostat
    g.add((tur_report_date_eurostat_instance, RDF.type, fibo_fnd_dt_fd.ExplicitDate))
    g.add((tur_report_date_eurostat_instance, fibo_fnd_dt_fd.hasDateValue, Literal('2021-04-13')))

    tur_report_date_GB_2020_instance = URIRef(MGKB + 'tur_gb_2020_report_date')  # total unemployment gb
    g.add((tur_report_date_GB_2020_instance, RDF.type, fibo_fnd_dt_fd.ExplicitDate))
    g.add((tur_report_date_GB_2020_instance, fibo_fnd_dt_fd.hasDateValue, Literal('2021-01-26')))

    for country, year_dict in total_ur.items():
        for year, rate in year_dict.items():
            tur_instance = URIRef(MGKB + 'tur_' + country + '_' + str(year))
            g.add((tur_instance, RDF.type, mgkb.TotalUnemploymentRate))
            g.add((tur_instance, fibo_fnd_utl_alx.hasArgument, total_unemployed_population))
            # data properties
            g.add((tur_instance, fibo_ind_ei_ei.hasIndicatorValue, Literal(rate)))  # hasIndicatorValue.
            g.add((tur_instance, schema.addressCountry, Literal(country)))  # schema.addressCountry
            g.add((tur_instance, dc.date, Literal(year)))  # dc.date
            if year == 2020 and country == 'GB':
                g.add((tur_instance, prov.wasGeneratedBy, ONS))
                g.add((tur_instance, fibo_fnd_arr_rep.hasReportDate, tur_report_date_GB_2020_instance))
            else:
                g.add((tur_instance, prov.wasGeneratedBy, eurostat))
                g.add((tur_instance, fibo_fnd_arr_rep.hasReportDate, tur_report_date_eurostat_instance))

    ### total unemployment rate report date instances
    yur_report_date_eurostat_instance = URIRef(MGKB + 'yur_eurostat_report_date')  # total unemployment eurostat
    g.add((yur_report_date_eurostat_instance, RDF.type, fibo_fnd_dt_fd.ExplicitDate))
    g.add((yur_report_date_eurostat_instance, fibo_fnd_dt_fd.hasDateValue, Literal('2021-04-13')))

    yur_report_date_GB_2020_instance = URIRef(MGKB + 'yur_gb_2020_report_date')  # total unemployment gb
    g.add((yur_report_date_GB_2020_instance, RDF.type, fibo_fnd_dt_fd.ExplicitDate))
    g.add((yur_report_date_GB_2020_instance, fibo_fnd_dt_fd.hasDateValue, Literal('2021-04-20')))

    for country, year_dict in youth_ur.items():
        for year, rate in year_dict.items():
            yur_instance = URIRef(MGKB + 'yur_' + country + '_' + str(year))
            g.add((yur_instance, RDF.type, mgkb.YouthUnemploymentRate))
            g.add((yur_instance, fibo_fnd_utl_alx.hasArgument, youth_unemployed_population))  # hasArgument
            g.add((yur_instance, fibo_ind_ei_ei.hasIndicatorValue, Literal(rate)))  # hasIndicatorValue
            g.add((yur_instance, schema.addressCountry, Literal(country)))  # schema.addressCountry
            g.add((yur_instance, dc.date, Literal(year)))  # dc.date
            if year == 2020 and country == 'GB':
                g.add((yur_instance, prov.wasGeneratedBy, UKParliament))
                g.add((yur_instance, fibo_fnd_arr_rep.hasReportDate, yur_report_date_GB_2020_instance))
            else:
                g.add((yur_instance, prov.wasGeneratedBy, eurostat))
                g.add((yur_instance, fibo_fnd_arr_rep.hasReportDate, yur_report_date_eurostat_instance))

    return rgdpr_dict, total_ur, youth_ur


def add_triples_for_one_tweet(g, row, entities_dict):
    row_dict = row.to_dict()

    idx = str(row['id'])
    # t_idx for ontology.
    t_idx = 't' + idx
    instance = URIRef(MGKB + t_idx)  # define the identifier for the instance of Post
    g.add((instance, RDF.type, sioc.Post))  # add instance of type Post
    g.add((instance, sioc.id, Literal(idx)))  # add sioc:id
    created_at = row['created_at']  # date created
    if created_at is not None:
        g.add((instance, dc.created, Literal(created_at)))  # add dc:created
    # u_id for ontology
    u_id_gen = row['author_id_gen']  # add userAccount
    # user_id
    u_idx = 'u' + u_id_gen
    user_instance = URIRef(MGKB + u_idx)
    # print(user_instance)
    g.add((user_instance, RDF.type, sioc.UserAccount))
    g.add((user_instance, sioc.id, Literal(u_id_gen)))  # userAccount has sioc:id.
    g.add((instance, sioc.has_creator, user_instance))  # has creator

    # place schema.
    p_id = 'p' + idx
    place_name = row['full_name']
    # print('place name:', place_name)
    country_code = row['country_code']
    twi_lat = row['lat']
    twi_lon = row['long']
    # pred_lat = row['pred_lat']
    # pred_lon = row['pred_lon']
    # consistent = row['consistent']
    place_instance = URIRef(MGKB + p_id)
    g.add((place_instance, RDF.type, schema.Place))  # place individual of schema:Place
    g.add((place_instance, schema.addressCountry, Literal(country_code)))  # place individual has country code

    # if not np.isnan(place_name):
    if place_name is not None:
        g.add((place_instance, sioc.name, Literal(place_name)))  # place individual has place_name

    if str(twi_lon) != 'nan' and str(twi_lat) != 'nan':
        g.add((place_instance, schema.latitude, Literal(twi_lat)))
        g.add((place_instance, schema.longitude, Literal(twi_lon)))
    # elif consistent:
    #     if str(pred_lat) != 'nan' and str(pred_lon) != 'nan':
    #         g.add((place_instance, schema.latitude, Literal(pred_lat)))
    #         g.add((place_instance, schema.longitude, Literal(pred_lon)))

    g.add((instance, schema.location, place_instance))  # has location

    # user mention instances.
    if str(row['user_mentions']) != 'nan':
        user_mentions = literal_eval(row['user_mentions'])
        user_mentions_id_dict = {'m' + str(idx) + '_' + str(mid): user_mention for mid, user_mention in
                                 enumerate(user_mentions)}
        # print(user_mentions_id_dict)

        for mid, user_mention in user_mentions_id_dict.items():
            user_mention_instance = URIRef(MGKB + mid)
            g.add((user_mention_instance, RDF.type, sioc.UserAccount))
            g.add((user_mention_instance, sioc.name, Literal(uuid.uuid4())))
            g.add((instance, schema.mentions, user_mention_instance))

    # Tag
    if str(row['hashtags']) != 'nan':
        hashtags = literal_eval(row['hashtags'])
        hashtags_dict = {'h' + str(idx) + '_' + str(hid): hashtag for hid, hashtag in
                         enumerate(hashtags)}
        # print(hashtags_dict)
        for hid, hashtag in hashtags_dict.items():
            hashtag_instance = URIRef(MGKB + hid)
            g.add((hashtag_instance, RDF.type, sioc_t.Tag))
            g.add((hashtag_instance, rdfs.label, Literal(hashtag)))
            g.add((instance, schema.mentions, hashtag_instance))  # rdfs:label

    # Topic.
    if not np.isnan(row['prim_topic']):
        prim_topic = row['prim_topic']
        g.add((instance, dc.subject, Literal(prim_topic)))

    # nee:Entity
    ent_mentions = [literal_eval(v) for e, v in row_dict.items() if e.startswith('entity_') if str(v) != 'nan']
    ### entity mention dict
    if len(ent_mentions) > 0:
        ents_mention_dict = {'em' + idx + '_' + str(entid): mention for entid, mention in enumerate(ent_mentions)}
        for entid, ent in ents_mention_dict.items():
            mention_instance = URIRef(MGKB + entid)
            # print('mention instance:', mention_instance)
            mention = ent['mention']  ## detectedAs.
            ent_idx = ent['id']
            rank_score = ent['score']
            url_ = entities_dict[str(ent_idx)]['url']  # hasMatchedURI
            # print('url: ', url_)
            g.add((mention_instance, RDF.type, nee.Entity))
            g.add((mention_instance, nee.hasMatchedURI, URIRef(url_)))
            g.add((mention_instance, nee.detectedAs, Literal(mention)))
            # get confidence of the mention.
            g.add((mention_instance, nee.confidence, Literal(rank_score)))
            g.add((instance, schema.mentions, mention_instance))

    # schema: interactionStatistics
    # if not np.isnan(row['like_count']):
    if row['like_count'] is not None:
        like_count = int(row['like_count'])
        like_instance = URIRef(MGKB + 'like' + idx)
        g.add((like_instance, RDF.type, schema.IneractionCounter))
        g.add((like_instance, schema.interactionType, schema.LikeAction))
        g.add((like_instance, schema.userInteractionCount, Literal(like_count)))
        g.add((instance, schema.interactionStatistics, like_instance))

    # if not np.isnan(row['retweet_count']):
    if row['retweet_count'] is not None:
        share_count = int(row['retweet_count'])
        share_instance = URIRef(MGKB + 'share' + idx)
        g.add((share_instance, RDF.type, schema.IneractionCounter))
        g.add((share_instance, schema.interactionType, schema.ShareAction))
        g.add((share_instance, schema.userInteractionCount, Literal(share_count)))
        g.add((instance, schema.interactionStatistics, share_instance))

    # if not np.isnan(row['reply_count']):
    if row['reply_count'] is not None:
        reply_count = int(literal_eval(row['reply_count']))
        # quote_count = int(row['quote_count'])
        reply_instance = URIRef(MGKB + 'reply' + idx)
        # quote_instance = URIRef(MGKB +'quote'+idx)
        g.add((reply_instance, RDF.type, schema.IneractionCounter))
        g.add((reply_instance, schema.interactionType, schema.ReplyAction))
        g.add((reply_instance, schema.userInteractionCount, Literal(reply_count)))
        g.add((instance, schema.interactionStatistics, reply_instance))

    ### sentiment
    senti_instance = URIRef(MGKB + 'senti' + idx)
    pred_sentiment = row['pred_sentiment']
    # pred_hatespeech = row['hatespeech_label']
    pred_hatespeech = row['hatespeech_pred']
    g.add((senti_instance, RDF.type, onyx.Emotion))

    if pred_sentiment == 0:
        g.add((senti_instance, onyx.hasEmotionCategory, negative_emotion))
    if pred_sentiment == 1:
        g.add((senti_instance, onyx.hasEmotionCategory, neutral_emotion))
    if pred_sentiment == 2:
        g.add((senti_instance, onyx.hasEmotionCategory, positive_emotion))

    if pred_hatespeech == 0:  # 'normal':
        g.add((senti_instance, onyx.hasEmotionCategory, normal_speech))
    if pred_hatespeech == 2:  # 'hatespeech':
        g.add((senti_instance, onyx.hasEmotionCategory, hate_speech))
    if pred_hatespeech == 1:  # 'offensive':
        g.add((senti_instance, onyx.hasEmotionCategory, offensive_speech))

    es_instance = URIRef(MGKB + 'es' + idx)
    g.add((es_instance, RDF.type, onyx.EmotionSet))
    g.add((es_instance, onyx.hasEmotion, senti_instance))
    g.add((instance, onyx.hasEmotionSet, es_instance))

    year = row['Year']
    # if country_code == 'GB':
    #     country_code = 'UK'

    # economic indicators
    if not np.isnan(year):
        tur_instance = URIRef(MGKB + 'tur_' + country_code + '_' + str(year))
        yur_instance = URIRef(MGKB + 'yur_' + country_code + '_' + str(year))
        rgdpr_instance = URIRef(MGKB + 'rgdpr_' + country_code + '_' + str(year))

        g.add((instance, fibo_fnd_rel_rel.isCharacterizedBy, tur_instance))
        g.add((instance, fibo_fnd_rel_rel.isCharacterizedBy, yur_instance))
        g.add((instance, fibo_fnd_rel_rel.isCharacterizedBy, rgdpr_instance))


if __name__ == '__main__':
    df = pd.read_csv('../data/kb/input/df_50_tm_sentiment_hsd_entities.csv', low_memory=False)
    df.dropna(subset=['created_at'], inplace=True)
    df.dropna(subset=['country_code'], inplace=True)
    df = df[df['topic_immig_max_score'] > 0.45]
    df['Year'] = [int(x) for x in df['created_at'].str.slice(stop=4)] #200k
    print(f'len of df: {len(df)}')  # 307339

    # df = df.sample(10)

    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    with open('../data/extracted/entities_dict_extracted_20210810.json') as file:
        entities_dict = json.load(file)

    ## entity resources
    define_entity_resources(entities_dict)
    rgdpr_dict, total_ur, youth_ur = define_economic_indicators()
    count = 0
    for idx, row in df.iterrows():
        print('processing ...', count, '\r')
        add_triples_for_one_tweet(g, row, entities_dict)
        count += 1

    g.serialize(destination=f"output/migrationsKB_{date_time}.nt", format="nt")
