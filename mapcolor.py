import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import numpy as np
import random
import re

class MapColoringVisualizer:
    def __init__(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#2d2d2d')

        # {node_id: {'pos': (x, y), 'connections': set(), 'color': color}}
        self.nodes = {}
        self.node_counter = 0

        self.node_radius = 0.8
        self.map_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
            '#FD79A8', '#A29BFE', '#6C5CE7', '#74B9FF', '#00B894'
        ]

        # drag state
        self.dragging_id = None
        self.drag_offset = (0.0, 0.0)

        # edge-edit state
        self.edit_edges_mode = False
        self.edge_select_first = None  # first node clicked in edge edit mode

        # heuristic toggles
        self.use_mrv = True
        self.use_degree = True

        self.setup_plot()
        self.setup_widgets()
        self._connect_events()  # drag + click handling

        plt.tight_layout()
        plt.show()

    # ---------------- UI / Plot ----------------

    def setup_plot(self):
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 9)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(True, alpha=0.1, color='white')
        subtitle = 'Map Coloring Problem Visualizer'
        if self.edit_edges_mode:
            subtitle += '  —  EDIT BORDERS: click two nodes to toggle'
        self.fig.suptitle(subtitle,
                          fontsize=22, color='white', fontweight='bold', y=0.94)

    def setup_widgets(self):
        # extra space for a third row of buttons (toggles)
        plt.subplots_adjust(bottom=0.34, top=0.88)

        button_color = '#4a4a4a'
        text_color = 'white'
        hover_color = '#6a6a6a'

        bh = 0.05
        bw = 0.13
        y0 = 0.22  # toggles row
        y1 = 0.16  # first row of buttons
        y2 = 0.10  # second row of buttons
        y3 = 0.04  # text inputs

        # Row 0: Heuristic Toggles
        ax_mrv = plt.axes([0.08, y0, bw, bh])
        ax_mrv.set_facecolor(button_color)
        self.mrv_btn = Button(ax_mrv, '', color=button_color, hovercolor=hover_color)
        self._style_btn(self.mrv_btn, text_color)
        self.mrv_btn.on_clicked(self.toggle_mrv)

        ax_degree = plt.axes([0.25, y0, bw, bh])
        ax_degree.set_facecolor(button_color)
        self.degree_btn = Button(ax_degree, '', color=button_color, hovercolor=hover_color)
        self._style_btn(self.degree_btn, text_color)
        self.degree_btn.on_clicked(self.toggle_degree)

        ax_both = plt.axes([0.42, y0, bw, bh])
        ax_both.set_facecolor(button_color)
        self.both_btn = Button(ax_both, '', color=button_color, hovercolor=hover_color)
        self._style_btn(self.both_btn, text_color)
        self.both_btn.on_clicked(self.toggle_both)

        # Row 1: Add, Auto Color, Clear All
        ax_add_btn = plt.axes([0.08, y1, bw, bh])
        ax_add_btn.set_facecolor(button_color)
        self.add_btn = Button(ax_add_btn, 'Add Region', color=button_color, hovercolor=hover_color)
        self._style_btn(self.add_btn, text_color)
        self.add_btn.on_clicked(self.add_node)

        ax_color_btn = plt.axes([0.25, y1, bw, bh])
        ax_color_btn.set_facecolor(button_color)
        self.color_btn = Button(ax_color_btn, 'Auto Color', color=button_color, hovercolor=hover_color)
        self._style_btn(self.color_btn, text_color)
        self.color_btn.on_clicked(self.auto_color)

        ax_clear_btn = plt.axes([0.42, y1, bw, bh])
        ax_clear_btn.set_facecolor(button_color)
        self.clear_btn = Button(ax_clear_btn, 'Clear All', color=button_color, hovercolor=hover_color)
        self._style_btn(self.clear_btn, text_color)
        self.clear_btn.on_clicked(self.clear_all)

        # Row 2: Edit Borders, Connect All, Clear Borders
        ax_edit_btn = plt.axes([0.08, y2, bw, bh])
        ax_edit_btn.set_facecolor(button_color)
        self.edit_btn = Button(ax_edit_btn, 'Edit Borders: OFF', color=button_color, hovercolor=hover_color)
        self._style_btn(self.edit_btn, text_color)
        self.edit_btn.on_clicked(self.toggle_edit_borders)

        ax_connect_all = plt.axes([0.25, y2, bw, bh])
        ax_connect_all.set_facecolor(button_color)
        self.connect_all_btn = Button(ax_connect_all, 'Connect All', color=button_color, hovercolor=hover_color)
        self._style_btn(self.connect_all_btn, text_color)
        self.connect_all_btn.on_clicked(self.connect_all)

        ax_clear_edges = plt.axes([0.42, y2, bw, bh])
        ax_clear_edges.set_facecolor(button_color)
        self.clear_edges_btn = Button(ax_clear_edges, 'Clear Borders', color=button_color, hovercolor=hover_color)
        self._style_btn(self.clear_edges_btn, text_color)
        self.clear_edges_btn.on_clicked(self.clear_borders)

        # Row 3: Inputs
        self.fig.text(0.08, y3 + 0.025, 'Enter borders for NEW region:', fontsize=11,
                      color=text_color, fontweight='bold')
        ax_connections = plt.axes([0.33, y3, 0.14, 0.04])
        ax_connections.set_facecolor(button_color)
        self.connections_box = TextBox(ax_connections, '', initial='1,2,3',
                                       color=button_color, hovercolor=hover_color)

        self.fig.text(0.50, y3 + 0.025, 'Set borders (pairs like 1-2,2-3):', fontsize=11,
                      color=text_color, fontweight='bold')
        ax_pairs = plt.axes([0.72, y3, 0.17, 0.04])
        ax_pairs.set_facecolor(button_color)
        self.pairs_box = TextBox(ax_pairs, '', initial='',
                                 color=button_color, hovercolor=hover_color)

        ax_set_pairs = plt.axes([0.90, y3, 0.08, 0.04])
        ax_set_pairs.set_facecolor(button_color)
        self.set_pairs_btn = Button(ax_set_pairs, 'Set Borders', color=button_color, hovercolor=hover_color)
        self._style_btn(self.set_pairs_btn, text_color, small=True)
        self.set_pairs_btn.on_clicked(self.set_borders_from_pairs)

        # Instruction box
        instruction_text = (
            'Build your map by adding regions and their borders.\n'
            '• Drag nodes to rearrange.\n'
            '• Toggle MRV / Degree heuristics above. "Both" toggles them together.\n'
            '• Click "Edit Borders" then click two nodes to toggle a border.\n'
            '• Use "Set Borders" with pairs like 1-2,2-3 to replace all borders.\n'
            '• "Connect All" builds a complete graph; "Clear Borders" removes all edges.\n'
            '• Auto Color runs backtracking with your selected heuristics + Forward Checking.'
        )
        self.fig.text(0.02, 0.92, instruction_text, fontsize=10, color='#cccccc', va='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#3a3a3a',
                      alpha=0.9, edgecolor='#5a5a5a', linewidth=1))

        self._refresh_toggle_labels()

    def _style_btn(self, btn, text_color, small=False):
        btn.label.set_color(text_color)
        btn.label.set_fontweight('bold')
        btn.label.set_fontsize(9 if small else 10)

    def _refresh_toggle_labels(self):
        self.mrv_btn.label.set_text(f"MRV: {'ON' if self.use_mrv else 'OFF'}")
        self.degree_btn.label.set_text(f"Degree: {'ON' if self.use_degree else 'OFF'}")
        both_state = 'ON' if (self.use_mrv and self.use_degree) else 'OFF'
        self.both_btn.label.set_text(f"Both: {both_state}")

    def draw_graph(self):
        """Draw the graph nodes/edges and overlay stats."""
        self.ax.clear()
        self.setup_plot()

        if not self.nodes:
            self.fig.canvas.draw_idle()
            return

        # Draw edges (once per pair)
        drawn_edges = set()
        for node_id, node_data in self.nodes.items():
            x1, y1 = node_data['pos']
            for connected_id in node_data['connections']:
                if connected_id in self.nodes:
                    edge = tuple(sorted([node_id, connected_id]))
                    if edge not in drawn_edges:
                        x2, y2 = self.nodes[connected_id]['pos']
                        self.ax.plot([x1, x2], [y1, y2], color='white',
                                     linewidth=4, alpha=0.8, solid_capstyle='round')
                        self.ax.plot([x1, x2], [y1, y2], color='#333333',
                                     linewidth=2, alpha=0.6, solid_capstyle='round')
                        drawn_edges.add(edge)

        # Draw nodes
        for node_id, node_data in self.nodes.items():
            x, y = node_data['pos']
            color = node_data['color']

            # selection highlight in edit mode
            if self.edit_edges_mode and self.edge_select_first == node_id:
                sel = patches.Circle((x, y), self.node_radius*1.15,
                                     facecolor='none', edgecolor='#FFD166',
                                     linewidth=3, zorder=2.5, linestyle='--')
                self.ax.add_patch(sel)

            # shadow
            shadow = patches.Circle((x + 0.1, y - 0.1), self.node_radius,
                                    facecolor='black', alpha=0.3, zorder=1)
            self.ax.add_patch(shadow)

            # outer + inner circle
            circle = patches.Circle((x, y), self.node_radius,
                                    facecolor=color, edgecolor='white',
                                    linewidth=3, zorder=2)
            self.ax.add_patch(circle)

            inner_circle = patches.Circle((x, y), self.node_radius * 0.7,
                                          facecolor=color, alpha=0.6, zorder=3)
            self.ax.add_patch(inner_circle)

            self.ax.text(x, y, str(node_id), ha='center', va='center',
                         fontsize=16, fontweight='bold', color='white',
                         zorder=4,
                         bbox=dict(boxstyle="circle,pad=0.1", facecolor='black', alpha=0.7))

        # Stats
        if len(self.nodes) > 1:
            used = len(set(nd['color'] for nd in self.nodes.values() if nd['color'] != '#888888'))
            stats_text = f"Regions: {len(self.nodes)}"
            if used > 0:
                stats_text += f" | Colors: {used}"
            if self.edit_edges_mode:
                stats_text += " | EDIT BORDERS ON"
            stats_text += f" | MRV={'ON' if self.use_mrv else 'OFF'}"
            stats_text += f" | Degree={'ON' if self.use_degree else 'OFF'}"
            self.ax.text(0.02, 0.98, stats_text,
                transform=self.ax.transAxes, fontsize=12, color='white',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='#4a4a4a',
                          alpha=0.9, edgecolor='#6a6a6a', linewidth=1))

        self.fig.canvas.draw_idle()

    # --------------- Graph editing ---------------

    def get_smart_position(self):
        """Sunflower (Vogel) layout centered on canvas."""
        n = self.node_counter  # 0-based for math
        if n == 0:
            return 5, 4
        phi = (np.sqrt(5) + 1) / 2
        angle = 2 * np.pi * n / (phi**2)
        radius = 0.6 * np.sqrt(n)
        x = 5 + radius * np.cos(angle)
        y = 4 + radius * np.sin(angle)
        # Clamp within bounds
        x = max(1, min(9, x))
        y = max(1, min(7, y))
        return x, y

    def add_node(self, event):
        """Add a new region; borders are taken from the input box as comma-separated ids."""
        self.node_counter += 1
        node_id = self.node_counter
        pos = self.get_smart_position()

        connections_text = self.connections_box.text.strip()
        connections = set()
        if connections_text:
            try:
                ids = [int(x.strip()) for x in connections_text.split(',') if x.strip()]
                connections = {cid for cid in ids if cid in self.nodes}
            except ValueError:
                connections = set()

        self.nodes[node_id] = {
            'pos': pos,
            'connections': connections,
            'color': '#888888'  # unassigned
        }
        # Make borders symmetric
        for cid in connections:
            self.nodes[cid]['connections'].add(node_id)

        # Reset placeholder in the box
        self.connections_box.set_val('1,2,3')

        self.draw_graph()
        print(f"Added region {node_id} with borders to regions {sorted(connections)}")

    def clear_all(self, event):
        self.nodes.clear()
        self.node_counter = 0
        self.edge_select_first = None
        self.draw_graph()
        print("Cleared all regions")

    # --------------- Border editing controls ---------------

    def toggle_edit_borders(self, event):
        self.edit_edges_mode = not self.edit_edges_mode
        self.edge_select_first = None
        self.edit_btn.label.set_text(f"Edit Borders: {'ON' if self.edit_edges_mode else 'OFF'}")
        self.draw_graph()

    def connect_all(self, event):
        ids = list(self.nodes.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                self.nodes[a]['connections'].add(b)
                self.nodes[b]['connections'].add(a)
        self.draw_graph()
        print("All regions connected (complete graph).")

    def clear_borders(self, event):
        for nid in self.nodes:
            self.nodes[nid]['connections'].clear()
        self.edge_select_first = None
        self.draw_graph()
        print("All borders cleared.")

    def set_borders_from_pairs(self, event):
        """Replace all borders with the pairs in pairs_box, format: 1-2,2-3,3-4"""
        text = self.pairs_box.text.strip()
        # Clear current borders
        for nid in self.nodes:
            self.nodes[nid]['connections'].clear()

        if text:
            pairs = re.split(r'[,\s]+', text)
            for p in pairs:
                if not p:
                    continue
                m = re.match(r'^(\d+)\s*-\s*(\d+)$', p)
                if not m:
                    continue
                a, b = int(m.group(1)), int(m.group(2))
                if a == b:
                    continue
                if a in self.nodes and b in self.nodes:
                    self.nodes[a]['connections'].add(b)
                    self.nodes[b]['connections'].add(a)

        self.edge_select_first = None
        self.draw_graph()
        print("Borders set from pairs.")

    # --------------- Toggles ---------------

    def toggle_mrv(self, event):
        self.use_mrv = not self.use_mrv
        self._refresh_toggle_labels()
        self.draw_graph()

    def toggle_degree(self, event):
        self.use_degree = not self.use_degree
        self._refresh_toggle_labels()
        self.draw_graph()

    def toggle_both(self, event):
        both_on = self.use_mrv and self.use_degree
        # if both currently ON, turn both OFF; else turn both ON
        new_state = not both_on
        self.use_mrv = new_state
        self.use_degree = new_state
        self._refresh_toggle_labels()
        self.draw_graph()

    # --------------- Drag & Click handling ---------------

    def _connect_events(self):
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)

    def _inside_node(self, nid, x, y):
        nx, ny = self.nodes[nid]['pos']
        return (x - nx)**2 + (y - ny)**2 <= (self.node_radius)**2

    def _find_node_at(self, x, y):
        # prefer topmost nodes (higher id) if overlapping
        for nid in sorted(self.nodes.keys(), reverse=True):
            if self._inside_node(nid, x, y):
                return nid
        return None

    def _clamp_pos(self, x, y):
        x = max(1, min(9, x))
        y = max(1, min(7, y))
        return x, y

    def _toggle_edge(self, a, b):
        if b in self.nodes[a]['connections']:
            self.nodes[a]['connections'].remove(b)
            self.nodes[b]['connections'].remove(a)
            print(f"Removed border {a}-{b}")
        else:
            self.nodes[a]['connections'].add(b)
            self.nodes[b]['connections'].add(a)
            print(f"Added border {a}-{b}")

    def _on_press(self, event):
        if event.inaxes != self.ax or not self.nodes:
            return
        if event.xdata is None or event.ydata is None:
            return

        nid = self._find_node_at(event.xdata, event.ydata)
        if nid is None:
            return

        if self.edit_edges_mode:
            if self.edge_select_first is None:
                self.edge_select_first = nid
                self.draw_graph()
            else:
                a, b = self.edge_select_first, nid
                if a != b:
                    self._toggle_edge(min(a,b), max(a,b))
                self.edge_select_first = None
                self.draw_graph()
            return

        # drag mode (not editing edges)
        cx, cy = self.nodes[nid]['pos']
        self.dragging_id = nid
        self.drag_offset = (event.xdata - cx, event.ydata - cy)

    def _on_motion(self, event):
        if self.dragging_id is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx, dy = self.drag_offset
        new_x = event.xdata - dx
        new_y = event.ydata - dy
        new_x, new_y = self._clamp_pos(new_x, new_y)
        self.nodes[self.dragging_id]['pos'] = (new_x, new_y)
        self.draw_graph()

    def _on_release(self, event):
        if self.dragging_id is None:
            return
        self.dragging_id = None
        self.drag_offset = (0.0, 0.0)
        self.draw_graph()

    # --------------- CSP core ---------------

    def _neighbors(self):
        return {nid: set(data['connections']) for nid, data in self.nodes.items()}

    def _initial_domains(self):
        """Full palette for uncolored nodes; respect any pre-colored nodes."""
        domains = {}
        for nid, data in self.nodes.items():
            if data['color'] != '#888888':
                domains[nid] = [data['color']]
            else:
                domains[nid] = list(self.map_colors)
        return domains

    def _select_next_var(self, assignment, domains, neighbors):
        """Select next variable to assign based on toggles."""
        unassigned = [v for v in self.nodes if v not in assignment]
        if not unassigned:
            return None

        if self.use_mrv and self.use_degree:
            # MRV + Degree tiebreaker
            def key(v):
                mrv = len(domains[v])
                deg_unassigned = sum((nb not in assignment) for nb in neighbors[v])
                return (mrv, -deg_unassigned, v)
            return min(unassigned, key=key)

        if self.use_mrv and not self.use_degree:
            # MRV only; tie-break by id
            def key(v):
                return (len(domains[v]), v)
            return min(unassigned, key=key)

        if self.use_degree and not self.use_mrv:
            # Degree only (no MRV): choose max degree among unassigned neighbors
            def key(v):
                deg_unassigned = sum((nb not in assignment) for nb in neighbors[v])
                # negative to get max with min()
                return (-deg_unassigned, v)
            return min(unassigned, key=key)

        # Neither: pick lowest id
        return min(unassigned)

    def _is_consistent(self, var, color, assignment, neighbors):
        for nb in neighbors[var]:
            if nb in assignment and assignment[nb] == color:
                return False
        return True

    def _forward_check(self, var, color, domains, neighbors):
        """Remove 'color' from neighbors' domains; returns list for undo. None => wipeout."""
        pruned = []
        for nb in neighbors[var]:
            if nb in domains and color in domains[nb]:
                if len(domains[nb]) == 1:
                    return None
                domains[nb].remove(color)
                pruned.append((nb, color))
        return pruned

    def _undo_pruned(self, pruned, domains):
        for v, c in pruned or []:
            if c not in domains[v]:
                domains[v].append(c)

    def _backtrack_color(self, assignment, domains, neighbors):
        if len(assignment) == len(self.nodes):
            return assignment

        var = self._select_next_var(assignment, domains, neighbors)
        if var is None:
            return assignment

        # (Optional LCV could be added here)
        for color in list(domains[var]):
            if not self._is_consistent(var, color, assignment, neighbors):
                continue
            assignment[var] = color
            pruned = self._forward_check(var, color, domains, neighbors)
            if pruned is not None:
                result = self._backtrack_color(assignment, domains, neighbors)
                if result is not None:
                    return result
            # undo
            del assignment[var]
            self._undo_pruned(pruned, domains)

        return None

    # --------------- Greedy fallback (degree order) ---------------

    def _greedy_degree_color(self):
        """Simple greedy by decreasing degree; used only if backtracking fails."""
        for node_data in self.nodes.values():
            node_data['color'] = '#888888'

        order = sorted(self.nodes.items(),
                       key=lambda x: len(x[1]['connections']),
                       reverse=True)
        for node_id, node_data in order:
            used = set()
            for nb in node_data['connections']:
                if nb in self.nodes:
                    used.add(self.nodes[nb]['color'])
            for color in self.map_colors:
                if color not in used:
                    node_data['color'] = color
                    break

    # --------------- Public: Auto Color button ---------------

    def auto_color(self, event):
        """Backtracking CSP with selectable MRV/Degree and forward checking (greedy fallback)."""
        if not self.nodes:
            return

        # reset visuals to unassigned gray
        for node_data in self.nodes.values():
            node_data['color'] = '#888888'

        neighbors = self._neighbors()
        domains = self._initial_domains()
        assignment = {}

        solution = self._backtrack_color(assignment, domains, neighbors)

        if solution is None:
            print("No valid coloring found with current palette; using greedy fallback.")
            self._greedy_degree_color()
            self.draw_graph()
            used = len(set(nd['color'] for nd in self.nodes.values()))
            print(f"Map colored using {used} colors (Greedy Degree Fallback).")
            return

        # Apply solution
        for nid, color in solution.items():
            self.nodes[nid]['color'] = color

        self.draw_graph()
        colors_used = len(set(solution.values()))
        mode = []
        if self.use_mrv: mode.append("MRV")
        if self.use_degree: mode.append("Degree")
        mode_label = " + ".join(mode) if mode else "No Heuristic"
        print(f"Map colored successfully using {colors_used} colors ({mode_label} + Forward Checking).")


# ---------------------- Run ----------------------

if __name__ == "__main__":
    print(" Map Coloring Visualizer")
    print("=" * 30)
    print("Instructions:")
    print("1. Click 'Add Region' to add a new territory")
    print("2. For new regions, you can enter comma-separated borders (e.g., '1,2,3')")
    print("3. Drag nodes to rearrange them on the canvas")
    print("4. Toggle MRV / Degree / Both to change the solver's variable selection")
    print("5. Click 'Edit Borders' then click two nodes to toggle their border")
    print("6. Use 'Set Borders' with pairs like '1-2,2-3' to replace all borders")
    print("7. 'Connect All' makes a complete graph; 'Clear Borders' removes all edges")
    print("8. Click 'Auto Color' to solve")
    print("\nStarting visualizer...")

    visualizer = MapColoringVisualizer()
