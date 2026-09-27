import pytest
import networkx as nx
import os
from routing.osm_network import validate_graph, load_graphml, save_graphml

def test_validate_valid_graph():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2, key=0, length=10)
    
    assert validate_graph(G) is True

def test_validate_empty_graph():
    G = nx.MultiDiGraph()
    with pytest.raises(ValueError, match="Graph contains no nodes"):
        validate_graph(G)

def test_validate_no_edges():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0, y=0)
    with pytest.raises(ValueError, match="Graph contains no edges"):
        validate_graph(G)

def test_validate_wrong_type():
    G = nx.Graph() # Not MultiDiGraph
    with pytest.raises(ValueError, match="Graph must be a NetworkX MultiDiGraph"):
        validate_graph(G)

def test_save_and_load_graphml(tmp_path):
    G = nx.MultiDiGraph()
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=1.0, y=1.0)
    G.add_edge(1, 2, key=0, length=14.14, name="Test Road")
    
    # Adding CRS to mock osmnx graph metadata requirement for saving sometimes
    G.graph['crs'] = 'epsg:4326'
    
    file_path = str(tmp_path / "test_graph.graphml")
    save_graphml(G, file_path)
    
    assert os.path.exists(file_path)
    
    G_loaded = load_graphml(file_path)
    assert len(G_loaded.nodes) == 2
    assert len(G_loaded.edges) == 1
    
    # Node keys might be loaded as strings by osmnx GraphML loader
    # Just verify structure is intact (u, v, data)
    assert list(G_loaded.edges(data=True))[0][2]['name'] == "Test Road"
