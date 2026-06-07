"""
Visualization service.
Generates multiple visualization formats from structured business process data:
- XMind mind map
- DrawIO flowchart
- Mermaid diagram
- ECharts interactive HTML
- Summary HTML dashboard
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path


def _ensure_output_dir(output_dir: str) -> str:
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ==================== Mermaid Generation ====================

def generate_mermaid(structured_data: Dict) -> str:
    """Generate Mermaid flowchart syntax from structured data."""
    lines = ["graph TD"]

    def _sanitize(text: str, max_len: int = 50) -> str:
        """Sanitize text for Mermaid - aggressively remove all special chars."""
        if not text or not text.strip():
            return "未命名"
        # Remove ALL Mermaid-sensitive characters
        for ch in ['"', '|', '(', ')', '[', ']', '{', '}', '<', '>', '&', '#', ';']:
            text = text.replace(ch, ' ')
        text = text.replace("\n", " ").replace("\\n", " ").strip()
        text = ' '.join(text.split())  # Collapse whitespace
        if not text:
            return "未命名"
        if len(text) > max_len:
            text = text[:max_len - 3] + "..."
        return text

    def _sanitize_edge(text: str, max_len: int = 30) -> str:
        """Sanitize edge labels aggressively."""
        return _sanitize(text, max_len)

    def _process_node(node: Dict, parent_id: str = "", depth: int = 0) -> List[str]:
        if not node or not isinstance(node, dict):
            return []
        result = []
        node_id = f"n{uuid.uuid4().hex[:8]}"
        title = _sanitize(node.get("title", ""), 50)
        node_type = node.get("type", "process")

        # Node shape based on type
        if node_type == "question":
            shape = f'{node_id}{{"{title}"}}'
        elif node_type == "decision":
            shape = f'{node_id}{{{{"{title}"}}}}'
        elif node_type == "script":
            shape = f'{node_id}["{title}"]'
        elif node_type == "action":
            shape = f'{node_id}[("{title}")]'
        elif node_type == "start":
            shape = f'{node_id}(("{title}"))'
        elif node_type == "end":
            shape = f'{node_id}(("{title}"))'
        else:
            shape = f'{node_id}["{title}"]'

        result.append(f"    {shape}")

        if parent_id:
            label = _sanitize_edge(node.get("condition", ""), 30)
            if label and label != "未命名":
                result.append(f"    {parent_id} -->|{label}| {node_id}")
            else:
                result.append(f"    {parent_id} --> {node_id}")

        # Process children
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                if child and isinstance(child, dict):
                    result.extend(_process_node(child, node_id, depth + 1))

        return result

    # Build from structured data
    root = structured_data.get("root") if isinstance(structured_data, dict) else None
    if root and isinstance(root, dict):
        mermaid_lines = _process_node(root)
        lines.extend(mermaid_lines)
    else:
        # Fallback: create a simple root node
        lines.append(f'    n0["业务流程"]')
        lines.append(f'    class n0 start')

    return "\n".join(lines)


# ==================== DrawIO Generation ====================

def generate_drawio_xml(structured_data: Dict) -> str:
    """Generate DrawIO XML from structured data."""
    cells = []
    cell_id = 2  # Start after root cells (0 and 1)
    x_pos = 40
    y_pos = 40
    level_width = 200
    node_height = 60
    level_gap = 80

    def _escape_xml(text: str) -> str:
        """Escape text for XML."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _add_node(label: str, x: int, y: int, w: int, h: int, style: str = "", parent_id: str = "") -> str:
        nonlocal cell_id
        cid = str(cell_id)
        cell_id += 1

        cell = f'''        <mxCell id="{cid}" value="{_escape_xml(label)}" style="{style}" vertex="1" parent="{parent_id or "1"}">
          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>
        </mxCell>'''
        return cid, cell

    def _add_edge(source: str, target: str, label: str = "", parent_id: str = "1") -> str:
        nonlocal cell_id
        cid = str(cell_id)
        cell_id += 1

        edge_style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
        if label:
            edge_style += f"label={_escape_xml(label)};"

        cell = f'''        <mxCell id="{cid}" value="{_escape_xml(label)}" style="{edge_style}" edge="1" parent="{parent_id}" source="{source}" target="{target}">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>'''
        return cid, cell

    def _process_node(node: Dict, px: int, py: int, depth: int, parent_cell_id: str = "") -> tuple:
        title = node.get("title", "未命名")
        node_type = node.get("type", "process")

        # Style based on node type
        if node_type == "question":
            style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;"
        elif node_type == "decision":
            style = "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;"
        elif node_type == "script":
            style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;"
        elif node_type == "action":
            style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;"
        elif node_type == "start":
            style = "ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;"
        elif node_type == "end":
            style = "ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;"
        else:
            style = "rounded=1;whiteSpace=wrap;html=1;"

        cid, cell = _add_node(title, px, py, 160, node_height, style)
        cells.append(cell)

        if parent_cell_id:
            cond = node.get("condition", "")
            _, edge = _add_edge(parent_cell_id, cid, cond)
            cells.append(edge)

        # Process children
        child_y = py + node_height + level_gap
        child_x = px
        children = node.get("children", [])
        for child in children:
            child_x, child_y = _process_node(child, child_x, child_y, depth + 1, cid)
            child_x += level_width

        return px, max(py, child_y)

    root = structured_data.get("root") if isinstance(structured_data, dict) else None
    if root and isinstance(root, dict):
        _process_node(root, x_pos, y_pos, 0)

    # Build full XML
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-01T00:00:00.000Z" agent="AI Business Process" version="21.0.0">
  <diagram name="业务流程" id="bp-diagram">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{chr(10).join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    return xml


# ==================== ECharts HTML Generation ====================

def generate_echarts_tree_html(structured_data: Dict) -> str:
    """Generate an interactive ECharts tree visualization as HTML."""
    # Convert structured data to ECharts tree format
    def _to_echarts_node(node: Dict) -> Dict:
        result = {
            "name": node.get("title", "未命名")[:50],
        }
        node_type = node.get("type", "")
        if node_type == "question":
            result["itemStyle"] = {"color": "#6c8ebf"}
        elif node_type == "decision":
            result["itemStyle"] = {"color": "#d6b656"}
        elif node_type == "script":
            result["itemStyle"] = {"color": "#82b366"}
        elif node_type == "action":
            result["itemStyle"] = {"color": "#9673a6"}

        children = node.get("children", [])
        if children:
            result["children"] = [_to_echarts_node(c) for c in children]
        return result

    root_node = structured_data.get("root") if isinstance(structured_data, dict) else None
    if not root_node or not isinstance(root_node, dict):
        root_node = {"title": "业务流程", "children": []}
    tree_data = _to_echarts_node(root_node)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>业务流程树形图</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 10px; }}
  #chart {{ width: 100%; height: 85vh; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ text-align: center; color: #333; font-size: 16px; margin: 6px 0; }}
  .tip {{ text-align: center; color: #888; font-size: 11px; margin-bottom: 6px; }}
</style>
</head>
<body>
<h1>业务流程树形图</h1>
<div class="tip">🖱 滚轮缩放 | 拖拽平移 | 点击节点展开/折叠</div>
<div id="chart"></div>
<script>
  var chart = echarts.init(document.getElementById('chart'));
  var data = {json.dumps(tree_data, ensure_ascii=False)};
  var option = {{
    tooltip: {{ trigger: 'item', triggerOn: 'mousemove', formatter: function(p) {{ return p.name; }} }},
    series: [{{
      type: 'tree',
      data: [data],
      top: '2%',
      left: '2%',
      bottom: '2%',
      right: '2%',
      symbolSize: 6,
      orient: 'TB',
      expandAndCollapse: true,
      initialTreeDepth: 2,
      roam: true,
      label: {{
        position: 'top',
        verticalAlign: 'middle',
        align: 'center',
        fontSize: 10,
        overflow: 'break',
        width: 100
      }},
      leaves: {{ label: {{ position: 'top', fontSize: 9, overflow: 'break', width: 80 }} }},
      emphasis: {{ focus: 'descendant', lineStyle: {{ width: 2 }} }},
      animationDuration: 300
    }}]
  }};
  chart.setOption(option);
  window.addEventListener('resize', function() {{ chart.resize(); }});
</script>
</body>
</html>'''

    return html


# ==================== Summary HTML Dashboard ====================

def generate_summary_html(
    structured_data: Dict,
    markdown: str,
    mermaid_syntax: str,
    stats: Optional[Dict] = None
) -> str:
    """Generate a summary HTML dashboard with all visualizations."""
    task_id = structured_data.get("task_id", "unknown")

    # Count stats
    def _count_nodes(node: Dict) -> int:
        count = 1
        for child in node.get("children", []):
            count += _count_nodes(child)
        return count

    root = structured_data.get("root") if isinstance(structured_data, dict) else None
    if not root or not isinstance(root, dict):
        root = {"title": "业务流程", "children": []}

    # Count branches
    def _count_branches(node: Dict) -> int:
        children = node.get("children", [])
        if len(children) > 1:
            return sum(1 + _count_branches(c) for c in children)
        elif len(children) == 1:
            return _count_branches(children[0])
        return 0

    total_branches = _count_branches(root) if root else 0
    total_nodes = _count_nodes(root) if root else 0

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>业务流程分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  .header {{ background: linear-gradient(135deg, #00a986 0%, #007a5e 100%); color: #fff; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
  .header p {{ opacity: 0.9; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .stat-card .value {{ font-size: 36px; font-weight: 700; color: #00a986; }}
  .stat-card .label {{ font-size: 14px; color: #888; margin-top: 4px; }}
  .section {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; overflow: hidden; }}
  .section-header {{ padding: 16px 20px; border-bottom: 1px solid #f0f0f0; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
  .section-body {{ padding: 20px; }}
  .mermaid {{ text-align: center; }}
  #echarts-tree {{ width: 100%; height: 600px; }}
  .flow-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
  .flow-stat-item {{ padding: 12px; background: #f8f9fa; border-radius: 6px; }}
  .flow-stat-item strong {{ color: #00a986; }}
  .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; background: #f0f0f0; padding: 4px; border-radius: 8px; }}
  .tab {{ padding: 8px 20px; border: none; background: none; cursor: pointer; border-radius: 6px; font-size: 14px; transition: all 0.2s; }}
  .tab.active {{ background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📊 业务流程分析报告</h1>
    <p>任务ID: {task_id} | 自动生成于 {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="value">{total_nodes}</div>
      <div class="label">流程节点总数</div>
    </div>
    <div class="stat-card">
      <div class="value">{total_branches}</div>
      <div class="label">分支路径数</div>
    </div>
    <div class="stat-card">
      <div class="value">{len(markdown)}</div>
      <div class="label">文档总字数</div>
    </div>
    <div class="stat-card">
      <div class="value">4</div>
      <div class="label">可视化类型</div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">📋 可视化视图</div>
    <div class="section-body">
      <div class="tabs">
        <button class="tab active" onclick="switchTab('mermaid')">Mermaid 流程图</button>
        <button class="tab" onclick="switchTab('echarts')">ECharts 树形图</button>
        <button class="tab" onclick="switchTab('markdown')">Markdown 文档</button>
      </div>
      <div id="tab-mermaid" class="tab-content active">
        <div class="mermaid">{mermaid_syntax}</div>
      </div>
      <div id="tab-echarts" class="tab-content">
        <div id="echarts-tree"></div>
      </div>
      <div id="tab-markdown" class="tab-content">
        <pre style="white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.6;">{markdown}</pre>
      </div>
    </div>
  </div>
</div>

<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'default' }});

  function switchTab(name) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab:nth-child(${{name === 'mermaid' ? 1 : name === 'echarts' ? 2 : 3}})`).classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
    if (name === 'echarts') {{
      setTimeout(function() {{
        var chart = echarts.init(document.getElementById('echarts-tree'));
        var data = {json.dumps(_to_echarts_node(root) if root else {}, ensure_ascii=False)};
        chart.setOption({{
          tooltip: {{ trigger: 'item' }},
          series: [{{ type: 'tree', data: [data], orient: 'LR', top: '2%', left: '5%', bottom: '2%', right: '15%',
            symbolSize: 8, label: {{ fontSize: 11 }}, expandAndCollapse: true, initialTreeDepth: -1, roam: true }}]
        }});
      }}, 100);
    }}
  }}
</script>
</body>
</html>'''

    return html


# ==================== Generate All ====================

def generate_all_visualizations(
    structured_data: Dict,
    markdown: str,
    output_dir: str = "./data/business_output",
    task_id: str = ""
) -> Dict[str, str]:
    """Generate all visualization formats and return paths."""
    _ensure_output_dir(output_dir)

    if not task_id:
        task_id = uuid.uuid4().hex[:12]

    results = {}

    # Mermaid
    mermaid_syntax = generate_mermaid(structured_data)
    mermaid_path = os.path.join(output_dir, f"{task_id}_mermaid.md")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(f"```mermaid\n{mermaid_syntax}\n```\n")
    results["mermaid"] = mermaid_path
    results["mermaid_syntax"] = mermaid_syntax

    # DrawIO
    drawio_xml = generate_drawio_xml(structured_data)
    drawio_path = os.path.join(output_dir, f"{task_id}.drawio")
    with open(drawio_path, "w", encoding="utf-8") as f:
        f.write(drawio_xml)
    results["drawio"] = drawio_path

    # ECharts HTML
    echarts_html = generate_echarts_tree_html(structured_data)
    echarts_path = os.path.join(output_dir, f"{task_id}_echarts.html")
    with open(echarts_path, "w", encoding="utf-8") as f:
        f.write(echarts_html)
    results["echarts"] = echarts_path

    # Summary HTML
    summary_html = generate_summary_html(structured_data, markdown, mermaid_syntax)
    summary_path = os.path.join(output_dir, f"{task_id}_summary.html")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_html)
    results["summary"] = summary_path

    # XMind file
    try:
        xmind_path = generate_xmind_file(structured_data, markdown, output_dir, task_id)
        if xmind_path:
            results["xmind"] = xmind_path
    except Exception as e:
        print(f"[Viz] XMind generation failed: {e}")

    return results


def generate_xmind_file(structured_data: Dict, markdown: str, output_dir: str, task_id: str) -> Optional[str]:
    """Generate an XMind-compatible file (ZIP with XML)."""
    import zipfile

    root = structured_data.get("root") if isinstance(structured_data, dict) else None
    if not root or not isinstance(root, dict):
        return None

    def _escape_xml(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _to_xmind_xml(node: Dict, topic_id: str = "") -> str:
        title = _escape_xml(node.get("title", "未命名")[:200])
        desc = node.get("description", "")
        condition = node.get("condition", "")
        tid = topic_id or "root"
        children_xml = ""
        # Add description and condition as child topics
        extra_children = ""
        if condition:
            extra_children += f'<topic id="{tid}_cond"><title>{_escape_xml("条件: " + condition[:200])}</title></topic>'
        if desc:
            extra_children += f'<topic id="{tid}_desc"><title>{_escape_xml("话术: " + desc[:500])}</title></topic>'
        children = node.get("children", [])
        if children or extra_children:
            children_xml = "<children><topics type=\"attached\">"
            if extra_children:
                children_xml += extra_children
            for i, child in enumerate(children):
                cid = f"{tid}_{i}"
                children_xml += _to_xmind_xml(child, cid)
            children_xml += "</topics></children>"
        return f'<topic id="{tid}"><title>{title}</title>{children_xml}</topic>'

    content_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<xmap-content xmlns="urn:xmind:xmap:xmlns:content:2.0" xmlns:fo="http://www.w3.org/1999/XSL/Format">
<sheet id="sheet1">
<topic id="root">
<title>{_escape_xml(root.get("title", "业务流程")[:100])}</title>
<children><topics type="attached">
{''.join(_to_xmind_xml(c, f"c{i}") for i, c in enumerate(root.get("children", [])))}
</topics></children>
</topic>
</sheet>
</xmap-content>'''

    manifest_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<manifest xmlns="urn:xmind:xmap:xmlns:manifest:1.0">
<file-entry full-path="content.xml" media-type="text/xml"/>
</manifest>'''

    xmind_path = os.path.join(output_dir, f"{task_id}.xmind")
    try:
        with zipfile.ZipFile(xmind_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('content.xml', content_xml.encode('utf-8'))
            zf.writestr('META-INF/manifest.xml', manifest_xml.encode('utf-8'))
        return xmind_path
    except Exception as e:
        print(f"[Viz] XMind ZIP creation failed: {e}")
        return None


def _to_echarts_node(node: Dict) -> Dict:
    """Convert structured data node to ECharts tree format."""
    result = {"name": node.get("title", "未命名")[:50]}
    node_type = node.get("type", "")
    colors = {
        "question": "#6c8ebf",
        "decision": "#d6b656",
        "script": "#82b366",
        "action": "#9673a6",
    }
    if node_type in colors:
        result["itemStyle"] = {"color": colors[node_type]}
    children = node.get("children", [])
    if children:
        result["children"] = [_to_echarts_node(c) for c in children]
    return result