import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import copy
import secrets
import os
import pickle
import glob
import torch.nn.functional as F
import torch_geometric as pyg

# TODO: refactor

class Evaluator:
    def __init__(self,
                 sampler,
                 discriminator,
                 dataset):
        self.sampler = sampler
        self.discriminator = discriminator
        self.dataset = dataset

    def probe(self, cls=None, discrete=False):
        graph = self.sampler(k=self.k, discrete=discrete)
        logits = self.discriminator(graph, edge_weight=graph.edge_weight)["logits"].mean(dim=0).tolist()
        if cls is not None:
            return logits[cls]
        return logits

    def detailed_probe(self):
        return pd.DataFrame(dict(
            logits_discrete=(ld := self.probe(discrete=True)),
            logits_continuous=(lc := self.probe(discrete=False)),
            prob_discrete=F.softmax(torch.tensor(ld), dim=0).tolist(),
            prob_continuous=F.softmax(torch.tensor(lc), dim=0).tolist(),
        ))

    @torch.no_grad()
    def predict(self, G, target_probs=None):
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True)])
        result = self.discriminator(batch)
        if target_probs is None or all([
            min_p <= result["probs"][0, classes].item() <= max_p
            for classes, (min_p, max_p) in target_probs.items()
        ]):
            return result
        return None

    @torch.no_grad()
    def quantatitive(self, sample_size=1000, sample_fn=None):
        sample_fn = sample_fn or (lambda: self.evaluate(bernoulli=True))
        p = []
        for i in range(1000):
            p.append(self.predict(sample_fn())["probs"][0].numpy().astype(float))
        return dict(label=list(self.dataset.GRAPH_CLS.values()),
                    mean=np.mean(p, axis=0),
                    std=np.std(p, axis=0))

    @torch.no_grad()
    def quantatitive_baseline(self, **kwargs):
        return self.quantatitive(sample_fn=lambda: nx.gnp_random_graph(n=self.sampler.n, p=1/self.sampler.n),
                                 **kwargs)

    # TODO: do not rely on dataset for drawing
    @torch.no_grad()
    def evaluate(self, *args, show=False, connected=False, **kwargs):
        self.sampler.eval()
        G = self.sampler.sample(*args, **kwargs)
        if connected:
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]
        if show:
            self.show(G)
            plt.show()
        return G

    def show(self, G, ax=None):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        pred = self.predict(G)
        logits = pred["logits"].mean(dim=0).tolist()
        probs = pred["probs"].mean(dim=0).tolist()
        print(f"{n=} {m=}")
        print(f"{logits=}")
        print(f"{probs=}")
        self.dataset.draw(G, ax=ax)

    def save_graph(self, G, cls_idx, root="results"):
        if isinstance(cls_idx, tuple):
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx[0]]}-{self.dataset.GRAPH_CLS[cls_idx[1]]}"
        else:
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx]}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        pickle.dump(G, open(f"{path}/{name}.pkl", "wb"))
        self.show(G)
        plt.savefig(f"{path}/{name}.png", bbox_inches="tight")
        plt.show()

    def load_graph(self, id, root="results"):
        path = f"{root}/{self.dataset.name}/*"
        G = pickle.load(open(glob.glob(f"{path}/{id}.pkl")[0], "rb"))
        self.show(G)
        return G

    def save_sampler(self, cls_idx, root="sampler_ckpts"):
        if isinstance(cls_idx, int):
            path = f"{root}/{self.dataset.name}/{cls_idx}"
        else:
            path = f"{root}/{self.dataset.name}/{'-'.join(map(str, cls_idx))}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        self.sampler.save(f"{path}/{name}.pt")

    def load_samplers(self, cls_idx, root="sampler_ckpts"):
        if isinstance(cls_idx, int):
            path = f"{root}/{self.dataset.name}/{cls_idx}"
        else:
            path = f"{root}/{self.dataset.name}/{'-'.join(map(str, cls_idx))}"
        for file in glob.glob(f"{path}/*.pt"):
            self.sampler.load(file)
            yield file
