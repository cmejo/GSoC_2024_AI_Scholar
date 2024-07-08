from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF, DC

# Create an RDF graph
g = Graph()
n = Namespace("http://example.org/")

# Add data to the graph
for doc in dataset:
    doc_uri = URIRef(f"http://example.org/document/{doc['id']}")
    g.add((doc_uri, DC.title, Literal(doc["title"])))
    g.add((doc_uri, DC.description, Literal(doc["text"])))
    for author in doc.get("authors", []):
        g.add((doc_uri, DC.creator, Literal(author)))

# Save the graph
g.serialize(destination="knowledge_graph.rdf", format="xml")

