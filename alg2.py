import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
from collections import deque

# Set Seaborn style
sns.set_style('whitegrid')

# Set page configuration
st.set_page_config(page_title="Algorithm Visualizer", layout="wide")
st.image("p1.png")
# Sidebar - Algorithm selection
st.sidebar.title("Algorithm Visualizer")
algorithm = st.sidebar.selectbox(
    "Select an Algorithm",
    (
        "Binary Search",
        "Merge Sort",
        "Quick Sort",
        "Binary to Decimal",
        "Breadth First Search (BFS)",
        "Depth First Search (DFS)",
        "Dijkstra's Algorithm",
        "Bellman-Ford Algorithm",
        "Floyd-Warshall Algorithm",
        "Kruskal's Algorithm",
        "Knapsack Problem",
        "Egyptian Fractions"
    )
)

st.title(f"{algorithm} Demonstration")

# Utility function to generate random array
def generate_random_array(size, min_val, max_val):
    return np.random.randint(min_val, max_val+1, size)

# Utility function to generate random graph
def generate_random_graph(num_nodes, num_edges, weighted=False, negative_weights=False):
    G = nx.gnm_random_graph(num_nodes, num_edges)
    if weighted:
        for (u, v) in G.edges():
            weight = random.randint(-10, 20) if negative_weights else random.randint(1, 20)
            G.edges[u, v]['weight'] = weight
    return G

# Binary Search Implementation
def binary_search_demo():
    st.sidebar.subheader("Parameters")
    array_size = st.sidebar.slider("Array Size", 5, 50, 10)
    target = st.sidebar.slider("Target Value", 0, 100, 50)
    
    array = sorted(generate_random_array(array_size, 0, 100))
    
    st.write(f"**Array:** {array}")
    st.write(f"**Target:** {target}")
    
    left, right = 0, len(array) - 1
    steps = []
    
    while left <= right:
        mid = (left + right) // 2
        steps.append(mid)
        if array[mid] == target:
            break
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(x=list(range(len(array))), y=array, palette="Blues", ax=ax)
    for i, step in enumerate(steps):
        ax.patches[step].set_color('red')
        st.pyplot(fig)
        if array[step] == target:
            st.success(f"Target found at index {step}")
            return
        else:
            ax.patches[step].set_color('gray')
    st.error("Target not found in the array.")

# Merge Sort Implementation
def merge_sort_demo():
    st.sidebar.subheader("Parameters")
    array_size = st.sidebar.slider("Array Size", 5, 50, 10)
    
    array = generate_random_array(array_size, 0, 100)
    st.write(f"**Unsorted Array:** {array}")
    
    def merge_sort(arr, depth=0):
        if len(arr) > 1:
            mid = len(arr) // 2
            L = arr[:mid]
            R = arr[mid:]
            
            merge_sort(L, depth + 1)
            merge_sort(R, depth + 1)
            
            i = j = k = 0
            
            # Visualization before merging
            st.write(f"{'  '*depth}Merging: {L} and {R}")
            
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            
            while i < len(L):
                arr[k] = L[i]
                i += 1
                k += 1
            
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1
            
            # Visualization after merging
            st.write(f"{'  '*depth}Result: {arr}")
    
    merge_sort(array.copy())
    st.write(f"**Sorted Array:** {array}")

# Quick Sort Implementation
def quick_sort_demo():
    st.sidebar.subheader("Parameters")
    array_size = st.sidebar.slider("Array Size", 5, 50, 10)
    
    array = generate_random_array(array_size, 0, 100)
    st.write(f"**Unsorted Array:** {array}")
    
    def quick_sort(arr, low, high, depth=0):
        if low < high:
            pi = partition(arr, low, high)
            st.write(f"{'  '*depth}Pivot at index {pi}: {arr}")
            quick_sort(arr, low, pi - 1, depth + 1)
            quick_sort(arr, pi + 1, high, depth + 1)
    
    def partition(arr, low, high):
        pivot = arr[high]
        i = low -1
        for j in range(low, high):
            if arr[j] <= pivot:
                i +=1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i+1
    
    quick_sort(array.copy(), 0, len(array) -1)
    st.write(f"**Sorted Array:** {array}")

# Binary to Decimal Conversion
def binary_to_decimal_demo():
    st.sidebar.subheader("Parameters")
    binary_str = st.sidebar.text_input("Binary Number", "1010")
    
    try:
        decimal = int(binary_str, 2)
        st.write(f"**Binary:** {binary_str}")
        st.write(f"**Decimal:** {decimal}")
    except ValueError:
        st.error("Invalid binary number.")

# Breadth First Search Implementation
def bfs_demo():
    st.sidebar.subheader("Parameters")
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, 10)
    num_edges = st.sidebar.slider("Number of Edges", num_nodes - 1, num_nodes*(num_nodes -1)//2, num_nodes)
    start_node = st.sidebar.slider("Start Node", 0, num_nodes - 1, 0)
    
    G = generate_random_graph(num_nodes, num_edges)
    visited = []
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            queue.extend(set(G.neighbors(node)) - set(visited))
    
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color='red', ax=ax)
    st.pyplot(fig)
    st.write(f"**BFS Traversal Order:** {visited}")

# Depth First Search Implementation
def dfs_demo():
    st.sidebar.subheader("Parameters")
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, 10)
    num_edges = st.sidebar.slider("Number of Edges", num_nodes - 1, num_nodes*(num_nodes -1)//2, num_nodes)
    start_node = st.sidebar.slider("Start Node", 0, num_nodes - 1, 0)
    
    G = generate_random_graph(num_nodes, num_edges)
    visited = []
    
    def dfs(node):
        if node not in visited:
            visited.append(node)
            for neighbor in G.neighbors(node):
                dfs(neighbor)
    
    dfs(start_node)
    
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color='green', ax=ax)
    st.pyplot(fig)
    st.write(f"**DFS Traversal Order:** {visited}")

# Dijkstra's Algorithm Implementation
def dijkstra_demo():
    st.sidebar.subheader("Parameters")
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, 6)
    num_edges = st.sidebar.slider("Number of Edges", num_nodes - 1, num_nodes*(num_nodes -1)//2, num_nodes)
    start_node = st.sidebar.slider("Start Node", 0, num_nodes - 1, 0)
    
    G = generate_random_graph(num_nodes, num_edges, weighted=True)
    path_lengths = nx.single_source_dijkstra_path_length(G, start_node)
    
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    st.pyplot(fig)
    
    st.write("**Shortest Path Lengths from Start Node:**")
    st.write(path_lengths)

# Bellman-Ford Algorithm Implementation
def bellman_ford_demo():
    st.sidebar.subheader("Parameters")
    num_nodes = st.sidebar.slider("Number of Nodes", 3, 10, 5)
    num_edges = st.sidebar.slider("Number of Edges", num_nodes - 1, num_nodes*(num_nodes -1), num_nodes)
    start_node = st.sidebar.slider("Start Node", 0, num_nodes - 1, 0)
    
    G = generate_random_graph(num_nodes, num_edges, weighted=True, negative_weights=True)
    try:
        path_lengths = nx.single_source_bellman_ford_path_length(G, start_node)
        
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color='lightcoral', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        st.pyplot(fig)
        
        st.write("**Shortest Path Lengths from Start Node:**")
        st.write(path_lengths)
    except nx.NetworkXUnbounded:
        st.error("Graph contains a negative weight cycle.")

# Floyd-Warshall Algorithm Implementation
def floyd_warshall_demo():
    st.sidebar.subheader("Parameters")
    num_nodes = st.sidebar.slider("Number of Nodes", 2, 10, 4)
    probability = st.sidebar.slider("Edge Creation Probability", 0.1, 1.0, 0.5)
    
    G = nx.gnp_random_graph(num_nodes, probability, directed=True)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)
    
    fw_result = dict(nx.floyd_warshall(G))
    
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color='skyblue', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    st.pyplot(fig)
    
    st.write("**All Pairs Shortest Path Lengths:**")
    st.write(pd.DataFrame(fw_result))

# Kruskal's Algorithm Implementation
def kruskal_demo():
    st.sidebar.subheader("Parameters")
    num_nodes = st.sidebar.slider("Number of Nodes", 3, 10, 5)
    num_edges = st.sidebar.slider("Number of Edges", num_nodes - 1, num_nodes*(num_nodes -1)//2, num_nodes)
    
    G = generate_random_graph(num_nodes, num_edges, weighted=True)
    mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightgrey', edge_color='grey', ax=ax)
    nx.draw_networkx_edges(mst, pos, edge_color='green', width=2, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    st.pyplot(fig)
    st.write("**Minimum Spanning Tree Edges:**")
    st.write(list(mst.edges(data=True)))

# Knapsack Problem Implementation
def knapsack_demo():
    st.sidebar.subheader("Parameters")
    num_items = st.sidebar.slider("Number of Items", 1, 20, 5)
    max_weight = st.sidebar.slider("Max Weight Capacity", 10, 100, 50)
    
    weights = np.random.randint(1, max_weight//2, num_items)
    values = np.random.randint(10, 100, num_items)
    
    st.write("**Items (Weight, Value):**")
    items = list(zip(weights, values))
    st.write(items)
    
    n = num_items
    W = max_weight
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    
    # Building the DP table
    for i in range(n +1):
        for w in range(W +1):
            if i ==0 or w ==0:
                K[i][w] =0
            elif weights[i-1] <= w:
                K[i][w] = max(values[i-1] + K[i-1][w - weights[i -1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    
    st.write(f"**Maximum Value:** {K[n][W]}")

# Egyptian Fractions Implementation
def egyptian_fractions_demo():
    st.sidebar.subheader("Parameters")
    numerator = st.sidebar.number_input("Numerator", min_value=1, value=4)
    denominator = st.sidebar.number_input("Denominator", min_value=2, value=13)
    
    def egyptian_fraction(num, den):
        ef = []
        while num != 0:
            x = (den // num) + 1
            ef.append(x)
            num = num*x - den
            den = den * x
        return ef
    
    ef = egyptian_fraction(numerator, denominator)
    st.write(f"**Egyptian Fraction Representation of {numerator}/{denominator}:**")
    fractions = [f"1/{x}" for x in ef]
    st.write(" + ".join(fractions))

# Main execution
if algorithm == "Binary Search":
    binary_search_demo()
elif algorithm == "Merge Sort":
    merge_sort_demo()
elif algorithm == "Quick Sort":
    quick_sort_demo()
elif algorithm == "Binary to Decimal":
    binary_to_decimal_demo()
elif algorithm == "Breadth First Search (BFS)":
    bfs_demo()
elif algorithm == "Depth First Search (DFS)":
    dfs_demo()
elif algorithm == "Dijkstra's Algorithm":
    dijkstra_demo()
elif algorithm == "Bellman-Ford Algorithm":
    bellman_ford_demo()
elif algorithm == "Floyd-Warshall Algorithm":
    floyd_warshall_demo()
elif algorithm == "Kruskal's Algorithm":
    kruskal_demo()
elif algorithm == "Knapsack Problem":
    knapsack_demo()
elif algorithm == "Egyptian Fractions":
    egyptian_fractions_demo()


st.info("build by dw 8-30-24")