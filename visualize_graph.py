import argparse
import json
import html as html_lib
from pathlib import Path


# ── colour palette per node type ──────────────────────────────────────────────
TYPE_COLORS = {
    "Character": "#4A90D9",
    "Event": "#6ABF69",
    "Location": "#E8943A",
    "Vehicle": "#9B59B6",
    "Object": "#95A5A6",
    "TimePoint": "#E74C3C",
    "Concept": "#F1C40F",
}

DEFAULT_COLOR = "#CCCCCC"


# ── HTML template ─────────────────────────────────────────────────────────────
def _html_page(title: str, nodes_json: str, edges_json: str, type_colors_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(title)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; }}
  svg {{ width: 100vw; height: 100vh; display: block; }}
  .link {{ stroke-opacity: 0.9; }}
  .link:hover {{ stroke-opacity: 1; }}
  .node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
  .node text {{ font-size: 11px; fill: #ccc; pointer-events: none; }}
  .tooltip {{ position: absolute; background: rgba(20,20,40,0.95); border: 1px solid #555;
              border-radius: 6px; padding: 10px 14px; font-size: 13px; pointer-events: none;
              max-width: 360px; line-height: 1.4; display: none; z-index: 100; }}
  .tooltip b {{ color: #fff; }}
  .tooltip .type-badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
                          font-size: 11px; margin-left: 6px; color: #fff; }}
  #legend {{ position: fixed; top: 12px; right: 12px; background: rgba(20,20,40,0.9);
             border: 1px solid #444; border-radius: 8px; padding: 12px 16px; font-size: 13px; }}
  #legend div {{ margin: 4px 0; }}
  #legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                  margin-right: 6px; vertical-align: middle; }}
  #title-bar {{ position: fixed; top: 12px; left: 12px; font-size: 18px; font-weight: 600;
                background: rgba(20,20,40,0.9); border: 1px solid #444; border-radius: 8px;
                padding: 8px 16px; }}
</style>
</head>
<body>
<div id="title-bar">{html_lib.escape(title)}</div>
<div id="legend"></div>
<div class="tooltip" id="tooltip"></div>
<svg></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const nodes = {nodes_json};
const links = {edges_json};
const typeColors = {type_colors_json};
const defaultColor = "{DEFAULT_COLOR}";

// Build legend
const types = [...new Set(nodes.map(n => n.type))].sort();
const legend = d3.select("#legend");
types.forEach(t => {{
  legend.append("div").html(`<span style="background:${{typeColors[t]||defaultColor}}"></span>${{t}} (${{nodes.filter(n=>n.type===t).length}})`);
}});

// Degree lookup from the rendered link set.
const degreeById = Object.fromEntries(nodes.map(n => [n.id, 0]));
for (const link of links) {{
  if (degreeById[link.source] != null) degreeById[link.source] += 1;
  if (degreeById[link.target] != null) degreeById[link.target] += 1;
}}

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("svg").attr("viewBox", [0, 0, width, height]);

// Zoom
const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => g.attr("transform", e.transform)));

// Build id lookup
const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));

// Force simulation
const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(120))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(30));

// Links
const link = g.append("g")
  .selectAll("line")
  .data(links)
  .join("line")
  .attr("class", "link")
  .attr("stroke", "#8fa4ff")
  .attr("stroke-width", d => {{
    const rel = String(d.relation || "").toLowerCase();
    if (rel === "performs" || rel === "undergoes" || rel === "experiences") return 2.8;
    if (rel === "occurs_at" || rel === "located_at" || rel === "present_on") return 2.2;
    return 1.8;
  }});

// Link labels
const linkLabel = g.append("g")
  .selectAll("text")
  .data(links)
  .join("text")
  .attr("font-size", 9)
  .attr("fill", "#888")
  .attr("text-anchor", "middle")
  .text(d => d.relation);

// Nodes
const node = g.append("g")
  .selectAll("g")
  .data(nodes)
  .join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end", (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }})
  );

node.append("circle")
  .attr("r", d => Math.max(7, Math.min(16, 6 + Math.sqrt((degreeById[d.id] || 0) + 1) * 3)))
  .attr("fill", d => typeColors[d.type] || defaultColor);

node.append("text")
  .attr("dx", 14)
  .attr("dy", 4)
  .text(d => d.name.length > 30 ? d.name.slice(0, 28) + "..." : d.name);

// Tooltip
const tooltip = d3.select("#tooltip");
node.on("mouseover", (e, d) => {{
  const color = typeColors[d.type] || defaultColor;
  let h = `<b>${{d.name}}</b><span class="type-badge" style="background:${{color}}">${{d.type}}</span><br>`;
  h += `<br><i>Degree:</i> ${{degreeById[d.id] || 0}}`;
  if (d.description) h += `<br>${{d.description}}`;
  if (d.aliases && d.aliases.length > 1) h += `<br><i>Aliases:</i> ${{d.aliases.join(", ")}}`;
  if (d.scene_refs && d.scene_refs.length) h += `<br><i>Scenes:</i> ${{d.scene_refs.join(", ")}}`;
  tooltip.html(h).style("display", "block");
}}).on("mousemove", (e) => {{
  tooltip.style("left", (e.pageX + 14) + "px").style("top", (e.pageY - 14) + "px");
}}).on("mouseout", () => tooltip.style("display", "none"));

link.on("mouseover", (e, d) => {{
  const src = typeof d.source === "object" ? d.source.name : d.source;
  const tgt = typeof d.target === "object" ? d.target.name : d.target;
  tooltip.html(`<b>${{src}}</b> &rarr; <i>${{d.relation}}</i> &rarr; <b>${{tgt}}</b>`).style("display", "block");
}}).on("mousemove", (e) => {{
  tooltip.style("left", (e.pageX + 14) + "px").style("top", (e.pageY - 14) + "px");
}}).on("mouseout", () => tooltip.style("display", "none"));

simulation.on("tick", () => {{
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel.attr("x", d => (d.source.x + d.target.x) / 2)
           .attr("y", d => (d.source.y + d.target.y) / 2);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>"""


def generate_visualizations(graph_path: str) -> None:
    graph_path = Path(graph_path)
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    movie_title = graph.get("title", "Unknown Movie")
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    viz_dir = graph_path.parent / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    type_colors_json = json.dumps(TYPE_COLORS)

    # ── collect all scene IDs ─────────────────────────────────────────────
    scene_ids = set()
    for n in nodes:
        for s in n.get("scene_refs", []):
            scene_ids.add(str(s))
    for e in edges:
        for s in e.get("scene_refs", []):
            scene_ids.add(str(s))

    # ── full graph page ───────────────────────────────────────────────────
    node_ids_in_edges = set()
    for e in edges:
        node_ids_in_edges.add(e["source"])
        node_ids_in_edges.add(e["target"])

    full_html = _html_page(
        title=f"{movie_title} — Full Graph",
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        type_colors_json=type_colors_json,
    )
    out = viz_dir / "full_graph.html"
    out.write_text(full_html, encoding="utf-8")
    print(f"  wrote {out}")

    # ── per-scene pages ───────────────────────────────────────────────────
    for scene_id in sorted(scene_ids):
        scene_nodes = [n for n in nodes if scene_id in [str(s) for s in n.get("scene_refs", [])]]
        scene_node_ids = {n["id"] for n in scene_nodes}
        scene_edges = []
        scene_context_ids = set(scene_node_ids)

        for e in edges:
            if scene_id not in [str(s) for s in e.get("scene_refs", [])]:
                continue
            scene_edges.append(e)
            if e["source"] not in scene_context_ids:
                scene_context_ids.add(e["source"])
            if e["target"] not in scene_context_ids:
                scene_context_ids.add(e["target"])

        scene_context_nodes = [n for n in nodes if n["id"] in scene_context_ids]

        scene_html = _html_page(
            title=f"{movie_title} — Scene {scene_id}",
            nodes_json=json.dumps(scene_context_nodes),
            edges_json=json.dumps(scene_edges),
            type_colors_json=type_colors_json,
        )
        out = viz_dir / f"scene_{scene_id}.html"
        out.write_text(scene_html, encoding="utf-8")
        print(f"  wrote {out}")

    # ── index page ────────────────────────────────────────────────────────
    index_links = [f'<li><a href="full_graph.html">Full Graph ({len(nodes)} nodes, {len(edges)} edges)</a></li>']
    for scene_id in sorted(scene_ids):
        sn = sum(1 for n in nodes if scene_id in [str(s) for s in n.get("scene_refs", [])])
        index_links.append(f'<li><a href="scene_{scene_id}.html">Scene {scene_id} ({sn} nodes)</a></li>')

    index_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_lib.escape(movie_title)} — Graph Visualizations</title>
<style>body{{font-family:system-ui;background:#1a1a2e;color:#eee;padding:40px;}}
a{{color:#4A90D9;}} li{{margin:8px 0;}}</style></head>
<body><h1>{html_lib.escape(movie_title)}</h1><h2>Knowledge Graph Visualizations</h2>
<ul>{"".join(index_links)}</ul></body></html>"""
    out = viz_dir / "index.html"
    out.write_text(index_html, encoding="utf-8")
    print(f"  wrote {out}")

    print(f"\nDone! Open {viz_dir / 'index.html'} in a browser.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a STAGE knowledge graph")
    parser.add_argument("--graph", required=True, help="Path to final_graph.json")
    args = parser.parse_args()
    generate_visualizations(args.graph)
