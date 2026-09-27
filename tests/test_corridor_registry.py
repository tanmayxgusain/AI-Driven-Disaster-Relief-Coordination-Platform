import pytest
import networkx as nx
from shapely.geometry import LineString
from routing.corridor_registry import (
    RoadCorridor,
    is_bridge,
    build_corridor_id,
    extract_geometry,
    build_corridor_registry
)

def test_is_bridge():
    assert is_bridge({'bridge': 'yes'}) is True
    assert is_bridge({'bridge': 'true'}) is True
    assert is_bridge({'bridge': '1'}) is True
    assert is_bridge({'bridge': ['no', 'yes']}) is True
    assert is_bridge({'bridge': True}) is True
    assert is_bridge({'bridge': 1}) is True
    
    assert is_bridge({'bridge': 'no'}) is False
    assert is_bridge({'bridge': 'false'}) is False
    assert is_bridge({'bridge': '0'}) is False
    assert is_bridge({'bridge': None}) is False
    assert is_bridge({}) is False

def test_build_corridor_id():
    assert build_corridor_id(1, 2, 0) == "edge:1:2:0"
    assert build_corridor_id("A", "B", "key") == "edge:A:B:key"

def test_extract_geometry():
    G = nx.MultiDiGraph()
    G.add_node(1, x=10.0, y=20.0)
    G.add_node(2, x=30.0, y=40.0)
    
    # 1. Fallback geometry
    geom = extract_geometry(1, 2, {}, G)
    assert isinstance(geom, LineString)
    assert list(geom.coords) == [(10.0, 20.0), (30.0, 40.0)]
    
    # 2. Existing geometry
    explicit_geom = LineString([(0, 0), (1, 1)])
    geom2 = extract_geometry(1, 2, {'geometry': explicit_geom}, G)
    assert geom2 == explicit_geom
    
    # 3. Missing node coords without explicit geom
    G.add_node(3)
    G.add_node(4)
    geom3 = extract_geometry(3, 4, {}, G)
    assert geom3 is None

def test_build_corridor_registry():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    
    # Road edge
    G.add_edge(1, 2, key=0, length=100.0, highway="primary", name="Main St")
    # Bridge edge (parallel)
    G.add_edge(1, 2, key=1, length=150.0, highway="secondary", bridge="yes")
    # Edge with no length (should be ignored)
    G.add_edge(1, 2, key=2, highway="tertiary")
    # Edge with invalid length (should be ignored)
    G.add_edge(1, 2, key=3, length=-10.0)
    
    registry = build_corridor_registry(G)
    
    assert len(registry) == 2
    
    id_road = build_corridor_id(1, 2, 0)
    id_bridge = build_corridor_id(1, 2, 1)
    
    assert id_road in registry
    assert id_bridge in registry
    
    assert registry[id_road].corridor_type == "road"
    assert registry[id_road].length_m == 100.0
    assert registry[id_road].highway == "primary"
    
    assert registry[id_bridge].corridor_type == "bridge"
    assert registry[id_bridge].length_m == 150.0


def test_build_corridor_registry_wrong_type():
    with pytest.raises(ValueError, match="Graph must be a networkx MultiDiGraph"):
        build_corridor_registry(nx.Graph())
