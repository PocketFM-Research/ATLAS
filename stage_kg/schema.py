"""
Graph schema definitions for STAGE knowledge graph.

Node types and relation types are grounded in the STAGE paper schema.
"""

from enum import Enum
from typing import Set, Tuple


class NodeType(str, Enum):
    CHARACTER = "Character"
    EVENT = "Event"
    LOCATION = "Location"
    TIME_POINT = "TimePoint"
    OBJECT = "Object"
    VEHICLE = "Vehicle"
    CONCEPT = "Concept"


class RelationType(str, Enum):
    # Event-Role relations (Table 7, paper)
    PERFORMS = "performs"           # entity actively executes or initiates an event
    UNDERGOES = "undergoes"         # entity is acted upon or targeted in an event
    EXPERIENCES = "experiences"     # character experiences internal mental/emotional event

    # Social relations
    KINSHIP_WITH = "kinship_with"       # kinship or marital relationships
    AFFINITY_WITH = "affinity_with"     # positive/cooperative relationships
    HOSTILITY_WITH = "hostility_with"   # negative/antagonistic relationships
    AFFILIATED_WITH = "affiliated_with" # belongs to or affiliated with an org/faction

    # Inter-Event relations
    PRECEDES = "precedes"           # one event occurs earlier than another
    CAUSES = "causes"               # one event directly causes another
    CONTRASTS_WITH = "contrasts_with"  # two events form a contrast or parallel
    REFERENCES = "references"       # one event refers to/recalls/describes another

    # Spatiotemporal relations
    OCCURS_AT = "occurs_at"         # Event -> Location
    OCCURS_ON = "occurs_on"         # Event -> TimePoint
    LOCATED_AT = "located_at"       # Character/Object/Concept -> Location
    PRESENT_ON = "present_on"       # Character/Object/Concept -> TimePoint

    # Object-related relations
    POSSESSES = "possesses"         # character/concept owns or holds an object
    USES = "uses"                   # character/object uses or operates another object

    # Semantic relations
    IS_A = "is_a"                   # type-instance or subclass -> Concept
    PART_OF = "part_of"             # part-whole or hierarchical relation


# Valid source -> relation -> target type triples
# Grounded in paper Table 7 and Table 12 (schema-constrained type repair examples)
VALID_TRIPLES: Set[Tuple[NodeType, RelationType, NodeType]] = {
    # Event-Role
    (NodeType.CHARACTER, RelationType.PERFORMS, NodeType.EVENT),
    (NodeType.CHARACTER, RelationType.UNDERGOES, NodeType.EVENT),
    (NodeType.CHARACTER, RelationType.EXPERIENCES, NodeType.EVENT),
    (NodeType.OBJECT, RelationType.PERFORMS, NodeType.EVENT),
    (NodeType.OBJECT, RelationType.UNDERGOES, NodeType.EVENT),
    (NodeType.CONCEPT, RelationType.PERFORMS, NodeType.EVENT),    # org initiates event
    (NodeType.VEHICLE, RelationType.PERFORMS, NodeType.EVENT),    # vehicle performs event
    (NodeType.VEHICLE, RelationType.UNDERGOES, NodeType.EVENT),  # vehicle acted upon

    # Social
    (NodeType.CHARACTER, RelationType.KINSHIP_WITH, NodeType.CHARACTER),
    (NodeType.CHARACTER, RelationType.AFFINITY_WITH, NodeType.CHARACTER),
    (NodeType.CHARACTER, RelationType.HOSTILITY_WITH, NodeType.CHARACTER),
    (NodeType.CHARACTER, RelationType.AFFILIATED_WITH, NodeType.CONCEPT),   # Table 12: Character->Organization
    (NodeType.CONCEPT, RelationType.AFFILIATED_WITH, NodeType.CONCEPT),

    # Inter-Event
    (NodeType.EVENT, RelationType.PRECEDES, NodeType.EVENT),
    (NodeType.EVENT, RelationType.CAUSES, NodeType.EVENT),
    (NodeType.EVENT, RelationType.CONTRASTS_WITH, NodeType.EVENT),
    (NodeType.EVENT, RelationType.REFERENCES, NodeType.EVENT),

    # Spatiotemporal — occurs_at is Event->Location; occurs_on is Event->TimePoint (Table 12)
    (NodeType.EVENT, RelationType.OCCURS_AT, NodeType.LOCATION),
    (NodeType.EVENT, RelationType.OCCURS_AT, NodeType.VEHICLE),           # event occurs aboard a vehicle
    (NodeType.EVENT, RelationType.OCCURS_ON, NodeType.TIME_POINT),         # Table 12: Event->TimePoint
    (NodeType.CHARACTER, RelationType.LOCATED_AT, NodeType.LOCATION),     # Table 12: Character->Location
    (NodeType.CHARACTER, RelationType.LOCATED_AT, NodeType.VEHICLE),      # character aboard vehicle
    (NodeType.OBJECT, RelationType.LOCATED_AT, NodeType.LOCATION),
    (NodeType.CONCEPT, RelationType.LOCATED_AT, NodeType.LOCATION),
    (NodeType.VEHICLE, RelationType.LOCATED_AT, NodeType.LOCATION),       # vehicle at a location
    (NodeType.CHARACTER, RelationType.PRESENT_ON, NodeType.TIME_POINT),
    (NodeType.OBJECT, RelationType.PRESENT_ON, NodeType.TIME_POINT),
    (NodeType.CONCEPT, RelationType.PRESENT_ON, NodeType.TIME_POINT),
    (NodeType.VEHICLE, RelationType.PRESENT_ON, NodeType.TIME_POINT),

    # Object-related
    (NodeType.CHARACTER, RelationType.POSSESSES, NodeType.OBJECT),
    (NodeType.CHARACTER, RelationType.POSSESSES, NodeType.VEHICLE),
    (NodeType.CONCEPT, RelationType.POSSESSES, NodeType.OBJECT),
    (NodeType.CHARACTER, RelationType.USES, NodeType.OBJECT),
    (NodeType.CHARACTER, RelationType.USES, NodeType.VEHICLE),
    (NodeType.OBJECT, RelationType.USES, NodeType.OBJECT),
    (NodeType.OBJECT, RelationType.PART_OF, NodeType.OBJECT),
    (NodeType.OBJECT, RelationType.PART_OF, NodeType.LOCATION),
    (NodeType.OBJECT, RelationType.PART_OF, NodeType.VEHICLE),
    (NodeType.LOCATION, RelationType.PART_OF, NodeType.LOCATION),
    (NodeType.VEHICLE, RelationType.PART_OF, NodeType.VEHICLE),
    (NodeType.VEHICLE, RelationType.PART_OF, NodeType.LOCATION),

    # Semantic
    (NodeType.CHARACTER, RelationType.IS_A, NodeType.CONCEPT),
    (NodeType.OBJECT, RelationType.IS_A, NodeType.CONCEPT),
    (NodeType.VEHICLE, RelationType.IS_A, NodeType.CONCEPT),
    (NodeType.CONCEPT, RelationType.IS_A, NodeType.CONCEPT),
    (NodeType.CHARACTER, RelationType.PART_OF, NodeType.CONCEPT),   # member of group
    (NodeType.CONCEPT, RelationType.PART_OF, NodeType.CONCEPT),
}


def is_valid_triple(
    src_type: NodeType, relation: RelationType, tgt_type: NodeType
) -> bool:
    """Return True if the (src_type, relation, tgt_type) triple is schema-valid."""
    return (src_type, relation, tgt_type) in VALID_TRIPLES


def get_valid_relations_for(
    src_type: NodeType, tgt_type: NodeType
) -> Set[RelationType]:
    """Return all relations valid between src_type and tgt_type."""
    return {
        r for (s, r, t) in VALID_TRIPLES if s == src_type and t == tgt_type
    }


NODE_TYPES = [t.value for t in NodeType]
RELATION_TYPES = [r.value for r in RelationType]
