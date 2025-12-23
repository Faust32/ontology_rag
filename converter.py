from rdflib import Graph
g = Graph()
g.parse("ontology.ttl", format="turtle")

g.serialize(destination='ontology.rdf', format='xml')
