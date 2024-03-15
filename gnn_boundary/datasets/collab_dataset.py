import networkx as nx
import pandas as pd
import torch_geometric as pyg


from .base_graph_dataset import BaseGraphDataset
from .utils import default_ax, unpack_G


class CollabDataset(BaseGraphDataset):

    NODE_CLS = {
        0: 'node'
    }

    GRAPH_CLS = {
        0: 'High Energy',
        1: 'Condensed Matter',
        2: 'Astro',
    }

    def __init__(self, *,
                 name='COLLAB',
                 url='https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/COLLAB.zip',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return ["COLLAB/COLLAB_A.txt",
                "COLLAB/COLLAB_graph_indicator.txt",
                "COLLAB/COLLAB_graph_labels.txt"]

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_zip(f'{self.raw_dir}/COLLAB.zip', self.raw_dir)

    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        graph_labels = pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) - 1
        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, 0, name='label')
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name='graph')
        return unpack_G(super_G)

    # TODO: use EDGE_WIDTH
    @default_ax
    def draw(self, G, pos=None, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               ax=ax,
                               nodelist=G.nodes,
                               node_size=500)
        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=6)

    def process(self):
        super().process()