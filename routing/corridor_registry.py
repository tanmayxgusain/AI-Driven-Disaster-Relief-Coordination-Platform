from typing import Optional, Dict, Literal, Any
from pydantic import BaseModel, ConfigDict
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString
import networkx as nx


class RoadCorridor(BaseModel):
    """
    A normalized deterministic representation of a road/bridge corridor graph edge.
    """
    corridor_id: str
    u: Any
    v: Any
    key: Any
    corridor_type: Literal["road", "bridge"]
    length_m: float
    geometry: Optional[Any] = None  # Holds Shapely geometries if available
    highway: Optional[Any] = None
    name: Optional[str] = None
    osmid: Optional[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

def is_bridge(edge_data: Dict[str, Any]) -> bool:
    """
    Deterministically determines if an OSM edge is a bridge.
    OSM 'bridge' tags are typically 'yes', 'true', '1'.
    """
    bridge_tag = edge_data.get('bridge', None)
    if bridge_tag is None:
        return False
        
    truthy_values = {"yes", "true", "1", "t"}
    
    if isinstance(bridge_tag, str):
        return bridge_tag.lower() in truthy_values
    elif isinstance(bridge_tag, list):
        return any(str(item).lower() in truthy_values for item in bridge_tag)
    elif isinstance(bridge_tag, bool):
        return bridge_tag
    elif isinstance(bridge_tag, int):
        return bridge_tag == 1
        
    return False

def build_corridor_id(u: Any, v: Any, key: Any) -> str:
    """Creates a deterministic string ID for an edge."""
    return f"edge:{u}:{v}:{key}"

def extract_geometry(u: Any, v: Any, edge_data: Dict[str, Any], graph: nx.MultiDiGraph) -> Optional[BaseGeometry]:
    """
    Extracts explicit geometry or constructs a fallback LineString from node coordinates.
    """
    if 'geometry' in edge_data and isinstance(edge_data['geometry'], BaseGeometry):
        return edge_data['geometry']
        
    # Fallback to straight line if node coords exist
    u_data = graph.nodes.get(u, {})
    v_data = graph.nodes.get(v, {})
    
    if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
        try:
            return LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])
        except Exception:
            return None
    return None

def build_corridor_registry(graph: nx.MultiDiGraph) -> Dict[str, RoadCorridor]:
    """
    Normalizes all eligible graph edges into deterministic corridor records.
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("Graph must be a networkx MultiDiGraph.")
        
    registry = {}
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Validate length
        length = data.get('length')
        if length is None:
            # Skip edges without an explicit length to avoid falsely assuming degrees as meters.
            continue
            
        try:
            length_m = float(length)
        except (ValueError, TypeError):
            continue
            
        if length_m <= 0:
            continue
            
        corridor_id = build_corridor_id(u, v, key)
        corridor_type = "bridge" if is_bridge(data) else "road"
        geom = extract_geometry(u, v, data, graph)
        
        # OSM tags can be single values or lists if edges were simplified
        highway = data.get('highway')
        name = data.get('name')
        if isinstance(name, list):
            name = ", ".join(str(n) for n in name)
            
        osmid = data.get('osmid')
        
        corridor = RoadCorridor(
            corridor_id=corridor_id,
            u=u,
            v=v,
            key=key,
            corridor_type=corridor_type,
            length_m=length_m,
            geometry=geom,
            highway=highway,
            name=name if isinstance(name, str) else str(name) if name else None,
            osmid=osmid
        )
        registry[corridor_id] = corridor
        
    return registry

