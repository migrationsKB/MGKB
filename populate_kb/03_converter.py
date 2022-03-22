from rdflib import Graph
import os


from rdflib import Graph
import os

g = Graph()
# for file in os.listdir("../output/mgkb2.0/"):
file = f"output/migrationsKB.xml"
outputfile = file.replace(".xml", ".nt")
print(file, "--->", outputfile)
g.parse(file)
g.serialize(destination=outputfile, format="nt")
