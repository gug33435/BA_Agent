import json
import copy
import uuid
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from collections import defaultdict, deque
from typing import Tuple, List, Dict, Any

# --- 1. Konfiguration und Konstanten ---
class LayoutConfig:
    TASK_WIDTH, TASK_HEIGHT = 100, 80
    GATEWAY_WIDTH, GATEWAY_HEIGHT = 40, 40
    EVENT_WIDTH, EVENT_HEIGHT = 30, 30
    
    HORIZONTAL_SPACING = 150
    VERTICAL_SPACING = 80
    LANE_PADDING_TOP = 60
    LANE_PADDING_BOTTOM = 60
    LANE_HEADER_WIDTH = 30
    LANE_CONTENT_PADDING_X = 60
    POOL_PADDING_X = 40
    POOL_PADDING_Y = 40
    ROUTING_MARGIN = 30

# --- 2. Hauptklasse: BPMNLayoutGenerator ---
class BPMNLayoutGenerator:
    """
    Hauptklasse zur Umwandlung einer JSON-Prozessbeschreibung in eine
    visuell gelayoutete und Signavio-spezifische BPMN 2.0 XML-Datei.
    """
    def __init__(self, process_json_data: dict):
        self.process_data = copy.deepcopy(process_json_data)
        self.nodes = {node['id']: node for node in self.process_data['nodes']}
        self.config = LayoutConfig()
        self.layout_info: Dict[str, Dict[str, Any]] = {node_id: {} for node_id in self.nodes}
        self.lane_layout: Dict[str, Dict[str, Any]] = {lane: {} for lane in self.process_data['akteure']}
        self.corridor_edges = set()
        self.ids = { "definitions": f"sid-{uuid.uuid4()}", "collaboration": f"sid-{uuid.uuid4()}", "process": f"sid-{uuid.uuid4()}", "participant": f"sid-{uuid.uuid4()}", "lane_set": f"sid-{uuid.uuid4()}", "diagram": f"sid-{uuid.uuid4()}", "plane": f"sid-{uuid.uuid4()}", "label_style_default": f"sid-{uuid.uuid4()}"}

    def _build_graph_representations(self):
        self.adj = defaultdict(list)
        self.rev_adj = defaultdict(list)
        self.node_lanes = {node_id: data['lane'] for node_id, data in self.nodes.items()}
        for node_id, node in self.nodes.items():
            for next_node_info in node.get('next_nodes', []):
                target_id = next_node_info.get('target_id') if isinstance(next_node_info, dict) else next_node_info
                if target_id in self.nodes:
                    self.adj[node_id].append(target_id)
                    self.rev_adj[target_id].append(node_id)

    def _optimize_gateway_lanes(self):
        for node_id, node in self.nodes.items():
            if 'gateway' in node['type'].lower():
                successors = self.adj.get(node_id, [])
                if len(successors) <= 1:
                    continue

                lane_counts = defaultdict(int)
                for succ_id in successors:
                    succ_lane = self.nodes[succ_id]['lane']
                    lane_counts[succ_lane] += 1
                
                if not lane_counts:
                    continue

                best_lane = max(lane_counts, key=lane_counts.get)
                max_count = lane_counts[best_lane]

                counts = list(lane_counts.values())
                if counts.count(max_count) > 1:
                    continue

                current_lane = node['lane']
                if current_lane != best_lane:
                    self.nodes[node_id]['lane'] = best_lane
                    self.node_lanes[node_id] = best_lane

    def _enforce_end_event_lanes(self):
        for node_id, node in self.nodes.items():
            if node['type'] == 'endEvent' and self.rev_adj[node_id]:
                predecessor_id = self.rev_adj[node_id][0]
                predecessor_lane = self.nodes[predecessor_id]['lane']
                if node['lane'] != predecessor_lane:
                    self.nodes[node_id]['lane'] = predecessor_lane
                    self.node_lanes[node_id] = predecessor_lane

    def _optimize_lane_order(self) -> List[str]:
        lane_crossings = defaultdict(int)
        for u, neighbors in self.adj.items():
            for v in neighbors:
                if self.node_lanes[u] != self.node_lanes[v]:
                    lane_crossings[(self.node_lanes[u], self.node_lanes[v])] += 1
        start_node_id = next((nid for nid, n in self.nodes.items() if n['type'] == 'startEvent'), None)
        start_lane = self.node_lanes[start_node_id] if start_node_id else list(self.process_data['akteure'])[0]
        ordered_lanes: List[str] = []
        remaining_lanes = set(self.process_data['akteure'])
        if start_lane in remaining_lanes:
            ordered_lanes.append(start_lane)
            remaining_lanes.remove(start_lane)
        
        current_lane = start_lane
        while remaining_lanes:
            best_candidate, max_score = None, -float('inf')
            for r_lane in remaining_lanes:
                score = lane_crossings.get((current_lane, r_lane), 0) * 2 - lane_crossings.get((r_lane, current_lane), 0)
                if score > max_score:
                    max_score, best_candidate = score, r_lane

            if not best_candidate:
                best_candidate = remaining_lanes.pop()
            else:
                remaining_lanes.remove(best_candidate)

            ordered_lanes.append(best_candidate)
            current_lane = best_candidate
        
        if ordered_lanes[0] != start_lane:
            ordered_lanes.remove(start_lane)
            ordered_lanes.insert(0, start_lane)

        return ordered_lanes

    def _calculate_node_positions(self, ordered_lanes: List[str]):
        max_rank = max(self.ranks.values()) if self.ranks else 0
        lane_y_cursor = self.config.POOL_PADDING_Y
        
        for i, lane_name in enumerate(ordered_lanes):
            nodes_in_lane = [nid for nid, n in self.nodes.items() if n['lane'] == lane_name]
            nodes_by_rank = defaultdict(list)
            for nid in nodes_in_lane: nodes_by_rank[self.ranks[nid]].append(nid)
            
            max_nodes_in_rank = max((len(nodes) for nodes in nodes_by_rank.values()), default=1)
            lane_height = max_nodes_in_rank * (self.config.TASK_HEIGHT + self.config.VERTICAL_SPACING) + self.config.LANE_PADDING_TOP + self.config.LANE_PADDING_BOTTOM
            self.lane_layout[lane_name].update({'y': lane_y_cursor, 'height': lane_height, 'order': i})
            
            for rank, nodes in sorted(nodes_by_rank.items()):
                y_spacing = lane_height / (len(nodes) + 1)
                for j, node_id in enumerate(sorted(nodes)):
                    width, height = self._get_node_dimensions(self.nodes[node_id]['type'])
                    x_pos = self.config.POOL_PADDING_X + self.config.LANE_HEADER_WIDTH + self.config.LANE_CONTENT_PADDING_X + rank * (self.config.TASK_WIDTH + self.config.HORIZONTAL_SPACING)
                    y_pos = lane_y_cursor + y_spacing * (j + 1) - (height / 2)
                    self.layout_info[node_id].update({'x': x_pos, 'y': y_pos, 'width': width, 'height': height})
            lane_y_cursor += lane_height
            
        self.pool_width = self.config.POOL_PADDING_X + self.config.LANE_HEADER_WIDTH + self.config.LANE_CONTENT_PADDING_X + (max_rank + 1.5) * (self.config.TASK_WIDTH + self.config.HORIZONTAL_SPACING)
        self.pool_height = lane_y_cursor

    def _assign_ranks(self):
        self.ranks = {nid: 0 for nid in self.nodes}
        in_degree = {nid: len(self.rev_adj[nid]) for nid in self.nodes}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                self.ranks[v] = max(self.ranks[v], self.ranks[u] + 1)
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)

        for u_id, u_node in self.nodes.items():
            is_diverging_gateway = 'gateway' in u_node['type'].lower() and len(self.adj[u_id]) > 1
            if is_diverging_gateway:
                successors = self.adj[u_id]
                if not successors: continue
                max_rank = max((self.ranks[v_id] for v_id in successors), default=self.ranks[u_id] + 1)
                for v_id in successors:
                    if self.ranks[v_id] < max_rank:
                        self.ranks[v_id] = max_rank

    def _resolve_cross_lane_collisions(self, ordered_lanes: List[str]):
        lane_orders = {lane_name: i for i, lane_name in enumerate(ordered_lanes)}
        made_adjustments = True
        iteration_guard = 0
        MAX_ITERATIONS = len(self.nodes) * len(self.nodes)
        while made_adjustments and iteration_guard < MAX_ITERATIONS:
            made_adjustments = False
            iteration_guard += 1
            for u_id in self.nodes:
                for v_id in self.adj[u_id]:
                    u_lane_order = lane_orders[self.nodes[u_id]['lane']]
                    v_lane_order = lane_orders[self.nodes[v_id]['lane']]
                    if abs(u_lane_order - v_lane_order) <= 1: continue
                    min_lane_ord, max_lane_ord = min(u_lane_order, v_lane_order), max(u_lane_order, v_lane_order)
                    colliding_nodes = []
                    for n_id in self.nodes:
                        if n_id in [u_id, v_id]: continue
                        n_lane_order = lane_orders[self.nodes[n_id]['lane']]
                        is_horizontally_between = self.ranks[u_id] <= self.ranks[n_id] < self.ranks[v_id]
                        is_vertically_between = min_lane_ord < n_lane_order < max_lane_ord
                        if is_vertically_between and is_horizontally_between:
                            colliding_nodes.append(n_id)
                    if colliding_nodes:
                        max_rank_collision = max(self.ranks[n_id] for n_id in colliding_nodes)
                        required_rank = max_rank_collision + 2
                        current_rank_v = self.ranks[v_id]
                        if current_rank_v < required_rank:
                            self.corridor_edges.add((u_id, v_id))
                            self.ranks[v_id] = required_rank
                            made_adjustments = True
                            q = deque([v_id])
                            visited = {v_id}
                            while q:
                                curr_id = q.popleft()
                                for succ_id in self.adj[curr_id]:
                                    if self.ranks[succ_id] <= self.ranks[curr_id]:
                                        self.ranks[succ_id] = self.ranks[curr_id] + 1
                                        if succ_id not in visited:
                                            visited.add(succ_id)
                                            q.append(succ_id)
                if made_adjustments:
                    break
            if made_adjustments:
                continue

    def _get_node_dimensions(self, node_type: str) -> Tuple[int, int]:
        if 'task' in node_type.lower(): return self.config.TASK_WIDTH, self.config.TASK_HEIGHT
        if 'gateway' in node_type.lower(): return self.config.GATEWAY_WIDTH, self.config.GATEWAY_HEIGHT
        if 'event' in node_type.lower(): return self.config.EVENT_WIDTH, self.config.EVENT_HEIGHT
        return self.config.TASK_WIDTH, self.config.TASK_HEIGHT

    def _calculate_all_edge_waypoints(self):
        for u_id, u_node in self.nodes.items():
            u_node['edges'] = []
            
            same_lane_successors = []
            if 'gateway' in u_node['type'].lower():
                for info in u_node.get('next_nodes', []):
                    target_id = info.get('id') or info.get('target_id')
                    if target_id and target_id in self.nodes and self.nodes[target_id]['lane'] == u_node['lane']:
                        same_lane_successors.append(info)

            for i, next_node_info in enumerate(u_node.get('next_nodes', [])):
                v_id = next_node_info.get('id') or next_node_info.get('target_id')
                if not v_id or v_id not in self.nodes: continue
                
                edge_label = next_node_info.get('label', '')
                u_layout, v_layout = self.layout_info[u_id], self.layout_info[v_id]
                v_node = self.nodes[v_id]
                rank_diff = self.ranks[v_id] - self.ranks[u_id]
                
                p_start = (u_layout['x'] + u_layout['width'], u_layout['y'] + u_layout['height'] / 2)
                p_end = (v_layout['x'], v_layout['y'] + v_layout['height'] / 2)
                exit_direction = 'right'
                
                u_lane_order = self.lane_layout[u_node['lane']]['order']
                v_lane_order = self.lane_layout[v_node['lane']]['order']

                if 'gateway' in u_node['type'].lower():
                    if u_lane_order != v_lane_order:
                        if v_lane_order < u_lane_order:
                            p_start, exit_direction = (u_layout['x'] + u_layout['width'] / 2, u_layout['y']), 'top'
                        else:
                            p_start, exit_direction = (u_layout['x'] + u_layout['width'] / 2, u_layout['y'] + u_layout['height']), 'bottom'
                    
                    elif len(same_lane_successors) >= 2:
                        try:
                            succ_ids = [s.get('id') or s.get('target_id') for s in same_lane_successors]
                            idx = succ_ids.index(v_id)
                            num_succ = len(same_lane_successors)
                            
                            if num_succ == 2:
                                p_start, exit_direction = ((u_layout['x'] + u_layout['width'] / 2, u_layout['y']), 'top') if idx == 0 else ((u_layout['x'] + u_layout['width'] / 2, u_layout['y'] + u_layout['height']), 'bottom')
                            elif num_succ == 3:
                                if idx == 0: p_start, exit_direction = (u_layout['x'] + u_layout['width'] / 2, u_layout['y']), 'top'
                                elif idx == 1: p_start, exit_direction = (u_layout['x'] + u_layout['width'], u_layout['y'] + u_layout['height'] / 2), 'right'
                                else: p_start, exit_direction = (u_layout['x'] + u_layout['width'] / 2, u_layout['y'] + u_layout['height']), 'bottom'
                            elif num_succ >= 4:
                                p_start, exit_direction = ((u_layout['x'] + u_layout['width'] / 2, u_layout['y']), 'top') if idx % 2 == 0 else ((u_layout['x'] + u_layout['width'] / 2, u_layout['y'] + u_layout['height']), 'bottom')
                        except ValueError:
                            pass

                waypoints = [p_start]
                
                if rank_diff > 1:
                    if 'gateway' in v_node['type'].lower() and not (u_id, v_id) in self.corridor_edges:
                        p_end = (v_layout['x'] + v_layout['width'] / 2, v_layout['y'])

                    if (u_id, v_id) in self.corridor_edges:
                        mid_x = p_start[0] + (v_layout['x'] - p_start[0]) / 2
                        waypoints.extend([(mid_x, p_start[1]), (mid_x, p_end[1])])
                    else:
                        routing_y = min(p_start[1], p_end[1]) - self.config.VERTICAL_SPACING
                        stub_x1 = p_start[0] + self.config.ROUTING_MARGIN
                        
                        waypoints.extend([
                            (stub_x1, p_start[1]),
                            (stub_x1, routing_y),
                            (p_end[0], routing_y)
                        ])
                
                elif rank_diff == 1:
                    if exit_direction == 'right':
                        mid_x = p_start[0] + self.config.ROUTING_MARGIN * 2
                        waypoints.extend([(mid_x, p_start[1]), (mid_x, p_end[1])])
                    else:
                        waypoints.append((p_start[0], p_end[1]))

                elif rank_diff < 0:
                    max_x = self.pool_width + self.config.POOL_PADDING_X
                    waypoints.extend([(max_x, p_start[1]), (max_x, p_end[1])])
                
                waypoints.append(p_end)
                u_node['edges'].append({'id': f"sid-{uuid.uuid4()}", 'target_id': v_id, 'waypoints': waypoints, 'label': edge_label})

    def _create_xml(self, ordered_lanes: List[str]) -> str:
        ns = {'': "http://www.omg.org/spec/BPMN/20100524/MODEL", 'bpmndi': "http://www.omg.org/spec/BPMN/20100524/DI", 'omgdc': "http://www.omg.org/spec/DD/20100524/DC", 'omgdi': "http://www.omg.org/spec/DD/20100524/DI", 'signavio': "http://www.signavio.com", 'xsi': "http://www.w3.org/2001/XMLSchema-instance"}
        root_attrs = { 'id': self.ids['definitions'], 'targetNamespace': ns['signavio'], **{f'xmlns:{k}' if k else 'xmlns': v for k, v in ns.items()}}
        definitions = Element('definitions', root_attrs)
        collaboration = SubElement(definitions, 'collaboration', {'id': self.ids['collaboration']})
        participant = SubElement(collaboration, 'participant', {'id': self.ids['participant'], 'name': self.process_data['prozessname'], 'processRef': self.ids['process']})
        process = SubElement(definitions, 'process', {'id': self.ids['process'], 'isExecutable': 'false'})
        laneset = SubElement(process, 'laneSet', {'id': self.ids['lane_set']})
        for lane_name in ordered_lanes:
            lane_id = f"sid-{uuid.uuid4()}"; self.lane_layout[lane_name]['id'] = lane_id
            lane = SubElement(laneset, 'lane', {'id': lane_id, 'name': lane_name})
            for nid in [nid for nid, n in self.nodes.items() if n['lane'] == lane_name]:
                SubElement(lane, 'flowNodeRef').text = nid
        for node_id, node in self.nodes.items():
            attrs = {'id': node_id, 'name': node.get('label', '')}
            if 'gateway' in node['type']: attrs['gatewayDirection'] = 'Diverging' if len(node.get('next_nodes',[])) > 1 else 'Converging'
            elem = SubElement(process, node['type'], attrs)
            for target_id in self.rev_adj[node_id]:
                edge = next((e for e in self.nodes[target_id].get('edges', []) if e['target_id'] == node_id), None)
                if edge: SubElement(elem, 'incoming').text = edge['id']
            for edge in node.get('edges', []): SubElement(elem, 'outgoing').text = edge['id']
        for u_id, u_node in self.nodes.items():
            for edge in u_node.get('edges', []):
                attrs = {'id': edge['id'], 'sourceRef': u_id, 'targetRef': edge['target_id']}
                if edge['label']: attrs['name'] = edge['label']
                SubElement(process, 'sequenceFlow', attrs)
        diagram = SubElement(definitions, 'bpmndi:BPMNDiagram', {'id': self.ids['diagram']})
        plane = SubElement(diagram, 'bpmndi:BPMNPlane', {'id': self.ids['plane'], 'bpmnElement': self.ids['collaboration']})
        pool_shape = SubElement(plane, 'bpmndi:BPMNShape', {'id': f"{self.ids['participant']}_gui", 'bpmnElement': self.ids['participant'], 'isHorizontal': 'true'})
        SubElement(pool_shape, 'omgdc:Bounds', {'x': str(self.config.POOL_PADDING_X), 'y': str(self.config.POOL_PADDING_Y), 'width': str(int(self.pool_width)), 'height': str(int(self.pool_height))})
        for lane_name, layout in self.lane_layout.items():
            lane_shape = SubElement(plane, 'bpmndi:BPMNShape', {'id': f"{layout['id']}_gui", 'bpmnElement': layout['id'], 'isHorizontal': 'true'})
            SubElement(lane_shape, 'omgdc:Bounds', {'x': str(self.config.POOL_PADDING_X + self.config.LANE_HEADER_WIDTH), 'y': str(int(layout['y'])), 'width': str(int(self.pool_width - self.config.LANE_HEADER_WIDTH)), 'height': str(int(layout['height']))})
        for node_id, layout in self.layout_info.items():
            shape = SubElement(plane, 'bpmndi:BPMNShape', {'id': f"{node_id}_gui", 'bpmnElement': node_id})
            SubElement(shape, 'omgdc:Bounds', {'x': str(int(layout['x'])), 'y': str(int(layout['y'])), 'width': str(layout['width']), 'height': str(layout['height'])})
            if self.nodes[node_id].get('label'):
                SubElement(shape, 'bpmndi:BPMNLabel', {'labelStyle': self.ids['label_style_default']})
        
        for node_id, node in self.nodes.items():
            for edge in node.get('edges', []):
                edge_shape = SubElement(plane, 'bpmndi:BPMNEdge', {'id': f"{edge['id']}_gui", 'bpmnElement': edge['id']})
                for p in edge['waypoints']: SubElement(edge_shape, 'omgdi:waypoint', {'x': str(round(p[0])), 'y': str(round(p[1]))})
                
                if edge['label']:
                    label_shape = SubElement(edge_shape, 'bpmndi:BPMNLabel', {'labelStyle': self.ids['label_style_default']})
                    if len(edge['waypoints']) > 1:
                        start_point = edge['waypoints'][0]
                        first_bend = edge['waypoints'][1]
                        
                        label_x = 0
                        label_y = 0
                        if 'gateway' in node['type'].lower():
                            if start_point[0] == first_bend[0]: 
                                label_x = first_bend[0] + 8
                                label_y = first_bend[1] + 15 if start_point[1] < first_bend[1] else first_bend[1] - 15
                            else:
                                label_x = start_point[0] + (first_bend[0] - start_point[0]) / 2
                                label_y = first_bend[1] - 20
                        else:
                            label_x = start_point[0] + 5
                            label_y = start_point[1] - 25
                        SubElement(label_shape, 'omgdc:Bounds', {'x': str(int(label_x)), 'y': str(int(label_y)), 'width': str(len(edge['label'])*7), 'height': '14'})
        style_default = SubElement(diagram, 'bpmndi:BPMNLabelStyle', {'id': self.ids['label_style_default']})
        SubElement(style_default, 'omgdc:Font', {'name': 'Arial', 'size': '12.0'})
        xml_str = tostring(definitions, 'utf-8')
        return minidom.parseString(xml_str).toprettyxml(indent="  ", encoding="UTF-8").decode('utf-8')

    def generate_bpmn_xml(self) -> str:
        self._build_graph_representations()
        self._optimize_gateway_lanes()
        ordered_lanes = self._optimize_lane_order()
        self._enforce_end_event_lanes()
        self._assign_ranks()
        self._resolve_cross_lane_collisions(ordered_lanes)
        self._calculate_node_positions(ordered_lanes)
        self._calculate_all_edge_waypoints()
        xml_output = self._create_xml(ordered_lanes)
        return xml_output

def generate_bpmn_xml_from_file(input_path: str) -> str:
    with open(input_path, 'r', encoding='utf-8') as f:
        process_json_data = json.load(f)
    return BPMNLayoutGenerator(process_json_data).generate_bpmn_xml()

if __name__ == "__main__":
    input_data = {
        "prozessziel": "Mitarbeiter bekommen ihre Auslagen schnell erstattet.", "prozessname": "Spesenabrechnung",
        "nodes": [
            {"type": "startEvent", "next_nodes": [{"id": "node_2"}], "id": "node_1", "lane": "Mitarbeiter", "label": "Spesen eingereicht"},
            {"type": "task", "next_nodes": [{"id": "node_3"}], "id": "node_2", "lane": "Buchhaltung", "label": "Abrechnung formal prüfen"},
            {"type": "exclusiveGateway", "next_nodes": [{"id": "node_4", "label": "Ja (< 100€)"}, {"id": "node_6", "label": "Nein (>= 100€)"}], "id": "node_3", "lane": "Buchhaltung", "label": "Summe < 100€?"},
            {"type": "task", "next_nodes": [{"id": "node_14"}], "id": "node_4", "lane": "HR-Tool", "label": "Zahlung automatisch freigeben"},
            {"type": "task", "next_nodes": [{"id": "node_7"}], "id": "node_5", "lane": "Teamleiter", "label": "Genehmigung durchführen"},
            {"type": "task", "next_nodes": [{"id": "node_5"}], "id": "node_6", "lane": "HR-Tool", "label": "Genehmigung beim Teamleiter anfordern"},
            {"type": "exclusiveGateway", "next_nodes": [{"id": "node_9", "label": "Genehmigt"}, {"id": "node_8", "label": "Abgelehnt"}], "id": "node_7", "lane": "Teamleiter", "label": "Abrechnung genehmigt?"},
            {"type": "task", "next_nodes": [{"id": "node_13"}], "id": "node_8", "lane": "HR-Tool", "label": "Antragsteller über Ablehnung informieren"},
            {"type": "task", "next_nodes": [{"id": "node_10"}], "id": "node_9", "lane": "Teamleiter", "label": "Genehmigung im System vermerken"},
            {"type": "task", "next_nodes": [{"id": "node_14"}], "id": "node_10", "lane": "Buchhaltung", "label": "Auszahlung manuell anstoßen"},
            {"type": "endEvent", "next_nodes": [], "id": "node_13", "lane": "HR-Tool", "label": "Prozess abgelehnt"},
            {"type": "exclusiveGateway", "next_nodes": [{"id": "node_15"}], "id": "node_14", "lane": "Buchhaltung", "label": ""},
            {"type": "endEvent", "next_nodes": [], "id": "node_15", "lane": "Mitarbeiter", "label": "Spesen erstattet"}
        ],
        "akteure": ["Mitarbeiter", "Buchhaltung", "HR-Tool", "Teamleiter"]
    }
    input_filename = "input.json"
    with open(input_filename, 'w', encoding='utf-8') as f: json.dump(input_data, f, indent=2, ensure_ascii=False)
    
    try:
        final_xml = BPMNLayoutGenerator(input_data).generate_bpmn_xml()
        output_filename = "output.bpmn"
        with open(output_filename, 'w', encoding='utf-8') as f: f.write(final_xml)
        print(f"\nErfolgreich! Die BPMN-Datei wurde in '{output_filename}' gespeichert.")
    except Exception as e:
        import traceback
        print(f"Ein Fehler ist aufgetreten: {e}")
        traceback.print_exc()