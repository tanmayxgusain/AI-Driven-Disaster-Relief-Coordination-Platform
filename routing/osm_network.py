import networkx as nx
import osmnx as ox
import os

def validate_graph(graph: nx.MultiDiGraph) -> bool:
    """
    Validates that the provided object is a usable MultiDiGraph.
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("Graph must be a NetworkX MultiDiGraph.")
        
    if len(graph.nodes) == 0:
        raise ValueError("Graph contains no nodes.")
        
    if len(graph.edges) == 0:
        raise ValueError("Graph contains no edges.")
        
    return True

def load_graphml(path: str) -> nx.MultiDiGraph:
    """
    Loads an OSMnx GraphML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GraphML file not found: {path}")
        
    graph = ox.load_graphml(path)
    validate_graph(graph)
    return graph

def save_graphml(graph: nx.MultiDiGraph, path: str) -> None:
    """
    Saves an OSMnx graph to a GraphML file.
    """
    validate_graph(graph)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ox.save_graphml(graph, filepath=path)

def download_drive_graph_from_place(place_name: str) -> nx.MultiDiGraph:
    """
    Downloads a drivable road network for a given place name using OSMnx.
    Note: Requires internet access. Should not be called in automated offline tests.
    """
    graph = ox.graph_from_place(place_name, network_type="drive")
    validate_graph(graph)
    return graph
