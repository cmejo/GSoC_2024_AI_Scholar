import networkx as nx
import matplotlib.pyplot as plt

def build_knowledge_graph(model, threshold=0.7):
    graph = nx.Graph()
    vocab = list(model.wv.index_to_key)
    
    for i, word1 in enumerate(vocab):
        for word2 in vocab[i+1:]:
            similarity = model.wv.similarity(word1, word2)
            if similarity > threshold:
                graph.add_edge(word1, word2, weight=similarity)
    
    return graph

def visualize_graph(graph):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, k=0.15)
    nx.draw_networkx(graph, pos, with_labels=True, node_size=50, font_size=10, edge_color='gray')
    plt.show()

def main():
    model = Word2Vec.load("word2vec.model")
    graph = build_knowledge_graph(model)
    visualize_graph(graph)

if __name__ == "__main__":
    main()
