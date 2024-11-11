import matplotlib.pyplot as plt
import networkx as nx

def visualization(G):
    """
    Displays the connected graph G  
    """
    # Compute the nodes positions with layout 
    pos = nx.spring_layout(G, k=0.15, iterations=50, weight=None)

    # Optionnal, adjust the size of the node
    node_sizes = [G.degree(node) * 1 for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color='skyblue',
            font_size=10, font_weight='bold', edge_color='gray', arrows=True)

    plt.title("Connected Graph of Articles and Links with Embeddings")
    plt.axis('off') 
    #plt.savefig("graph.png", dpi=1000)  
    plt.show()