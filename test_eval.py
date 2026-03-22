import json
from pathlib import Path

# Load the generated graph
graph_path = Path("path/to/your/graph.json")

with open(graph_path) as f:
    graph_data = json.load(f)

print(f"Nodes: {len(graph_data['nodes'])}")
print(f"Edges: {len(graph_data['edges'])}")

# Count character nodes + scene_refs
characters = [n for n in graph_data["nodes"] if n["type"] == "Character"]
print(f"\nCharacters: {len(characters)}")

for char in characters[:3]:
    print(f"  {char['id']}: {char['name']} - scenes: {char.get('scene_refs', [])}")

# Test claim extractor
from stage_kg.evaluation.claim_extractor import ClaimExtractor

extractor = ClaimExtractor()
test_text = "Kirk performed the famous scene in the shuttle bay."

claims = extractor.extract_claims(test_text, "1", "char_123")
print(f"\nExtracted {len(claims)} claims:")
for claim in claims[:5]:
    print(f"  S: {claim.subject}, P: {claim.predicate}, O: {claim.object}")

# Test dummy generator
def dummy_text_generator(graph_data, scene_id, character_id):
    char_name = None
    for node in graph_data.get("nodes", []):
        if node["id"] == character_id:
            char_name = node.get("name", "The character")
            break
    
    if not char_name:
        return "No description available."
    
    events = []
    for edge in graph_data.get("edges", []):
        if edge["source"] == character_id and edge["relation"] in ["performs", "experiences", "undergoes"]:
            for node in graph_data["nodes"]:
                if node["id"] == edge["target"] and node["type"] == "Event":
                    if scene_id in node.get("scene_refs", []):
                        events.append(node.get("name", "an event"))
            break
    
    if events:
        return f"{char_name} {', '.join(events)} in scene {scene_id}."
    else:
        return f"{char_name} appears in scene {scene_id}."

# Test with first character
if characters:
    char = characters[0]
    scenes = char.get("scene_refs", [])
    if scenes:
        generated_text = dummy_text_generator(graph_data, scenes[0], char["id"])
        print(f"\nGenerated text for {char['name']} in scene {scenes[0]}:")
        print(f"  '{generated_text}'")
        
        # Extract claims from generated text
        claims = extractor.extract_claims(generated_text, scenes[0], char["id"])
        print(f"  Extracted {len(claims)} claims from generated text")
