import json
import pprint
from datetime import datetime

import rdflib
from rdflib import Namespace
from rdflib import URIRef, BNode, Literal

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

# initialize a graph
g = rdflib.Graph()


# define namespace
sioc = Namespace('http://rdfs.org/sioc/ns#')
sioc_t = Namespace('http://rdfs.org/sioc/types#')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
nee = Namespace('http://www.ics.forth.gr/isl/oae/core#')
schema = Namespace('http://schema.org/')
onyx = Namespace('http://www.gsi.dit.upm.es/ontologies/onyx/ns#')
wna = Namespace('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#')
dc = Namespace('http://purl.org/dc/elements/1.1/')
fibo_ind_ei_ei= Namespace('https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/')
fibo_fnd_dt_fd=Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/DatesAndTimes/FinancialDates/')
fibo_fnd_rel_rel = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/')
fibo_fnd_utl_alx =Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Utilities/Analytics/')
fibo_fnd_arr_doc = Namespace('https://spec.edmcouncil.org/fibo/ontology/FND/Arrangements/Documents/')

## binding
g.bind("sioc", sioc)
g.bind("sioc_t", sioc_t)
g.bind("rdfs", rdfs)
g.bind("wna", wna)
g.bind("nee", nee)
g.bind("dc", dc)
g.bind("schema", schema)
g.bind("onyx",onyx)
g.bind("fibo-ind-ei-ei", fibo_ind_ei_ei)
g.bind("fibo-fnd-dt-fd", fibo_fnd_dt_fd)
g.bind("fibo-fnd-rel-rel", fibo_fnd_rel_rel)
g.bind("fibo-fnd-utl-alx", fibo_fnd_utl_alx)
g.bind("fibo-fnd-arr-doc", fibo_fnd_arr_doc)

### populate
neutral_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#neutral-emotion')
negative_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#negative-emotion')
positive_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#positive-emotion')

### define post and emotion
g.add((sioc.Post, onyx.hasEmotionSet, onyx.EmotionSet))
g.add((onyx.EmotionSet, onyx.hasEmotion, onyx.Emotion))
g.add((onyx.Emotion, onyx.hasEmotionCategory, neutral_emotion))
g.add((onyx.Emotion, onyx.hasEmotionCategory, negative_emotion))
g.add((onyx.Emotion, onyx.hasEmotionCategory, positive_emotion))

## define post and interactionCounter
g.add((sioc.Post, schema.interactionStatistics, schema.InteractionCounter))
g.add((schema.InteractionCounter, schema.interactionType, schema.LikeAction ))
g.add((schema.InteractionCounter, schema.interactionType, schema.ShareAction ))
g.add((schema.InteractionCounter, schema.interactionType, schema.ReplyAction ))


### post and location
g.add((sioc.Post, schema.location, schema.Place))

## post and UserAccount
# creator
g.add((sioc.Post, sioc.has_creator,sioc.UserAccount))
# mentions
g.add((sioc.Post, schema.mentions, sioc.UserAccount))

### post and unemployment rate
g.add((sioc.Post, fibo_fnd_rel_rel.isCharacterizedBy, fibo_ind_ei_ei.UnemploymentRate))
g.add((fibo_ind_ei_ei.UnemploymentRate, fibo_fnd_utl_alx.hasArgument, fibo_ind_ei_ei.UnemployedPopulation))
g.add((fibo_ind_ei_ei.UnemploymentRate, fibo_fnd_arr_doc.hasReportingPeriod, fibo_fnd_dt_fd.ExplicitDatePeriod))

### post and GrossDomesticProduct rate
g.add((sioc.Post, fibo_fnd_rel_rel.isCharacterizedBy, fibo_ind_ei_ei.GrossDomesticProduct))
g.add((fibo_ind_ei_ei.GrossDomesticProduct, fibo_fnd_arr_doc.hasReportingPeriod, fibo_fnd_dt_fd.ExplicitDatePeriod))


### post and entity mentions
g.add((sioc.Post, schema.mentions, nee.Entity))
g.add((nee.Entity, nee.hasMatchedURI, rdfs.Resource))

### post and topic
g.add((sioc.Post, schema.mentions, sioc_t.Tag))

g.serialize(f"output/migrationsKB_{date_time}.rdf", format="application/rdf+xml")