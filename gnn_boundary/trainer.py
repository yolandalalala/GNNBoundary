import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm, trange
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
# from .datasets import *


class Trainer:
    def __init__(self,
                 sampler,
                 discriminator,
                 criterion,
                 scheduler,
                 optimizer,
                 dataset,
                 budget_penalty=None,):
        self.sampler = sampler
        self.discriminator = discriminator
        self.criterion = criterion
        self.budget_penalty = budget_penalty
        self.scheduler = scheduler
        self.optimizer = optimizer if isinstance(optimizer, list) else [optimizer]
        self.dataset = dataset
        self.iteration = 0

    def init(self):
        self.sampler.init()
        self.iteration = 0

    def train(self, iterations,
              show_progress=True,
              target_probs: dict[int, tuple[float, float]] = None,
              target_size=None,
              w_budget_init=1,
              w_budget_inc=1.05,
              w_budget_dec=0.99,
              k_samples=32):
        # self.bkup_state = copy.deepcopy(self.sampler.state_dict())
        # self.bkup_criterion = copy.deepcopy(self.criterion)
        # self.bkup_iteration = self.iteration
        self.discriminator.eval()
        self.sampler.train()
        budget_penalty_weight = w_budget_init
        for _ in (bar := tqdm(
            range(int(iterations)),
            initial=self.iteration,
            total=self.iteration+iterations,
            disable=not show_progress
        )):
            for opt in self.optimizer:
                opt.zero_grad()
            cont_data = self.sampler(k=k_samples, mode='continuous')
            disc_data = self.sampler(k=1, mode='discrete', expected=True)
            # TODO: potential bug
            cont_out = self.discriminator(cont_data, edge_weight=cont_data.edge_weight)
            disc_out = self.discriminator(disc_data, edge_weight=disc_data.edge_weight)
            if target_probs and all([
                min_p <= disc_out["probs"][0, classes].item() <= max_p
                for classes, (min_p, max_p) in target_probs.items()
            ]):
                if target_size and self.sampler.expected_m <= target_size:
                    break
                budget_penalty_weight *= w_budget_inc
            else:
                budget_penalty_weight = max(w_budget_init, budget_penalty_weight * w_budget_dec)

            loss = self.criterion(cont_out | self.sampler.to_dict())
            if self.budget_penalty:
                loss += self.budget_penalty(self.sampler.theta) * budget_penalty_weight
            loss.backward()  # Back-propagate gradients

            for opt in self.optimizer:
                opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # logging
            size = self.sampler.expected_m
            scores = disc_out["logits"].mean(axis=0).tolist()
            score_dict = {v: scores[k] for k, v in self.dataset.GRAPH_CLS.items()}
            penalty_weight = {'bpw': budget_penalty_weight} if self.budget_penalty else {}
            bar.set_postfix({'size': size} | penalty_weight | score_dict)
            # print(f"{iteration=}, loss={loss.item():.2f}, {size=}, scores={score_dict}")
            self.iteration += 1
        else:
            return False
        return True

    @torch.no_grad()
    def predict(self, G):
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True)])
        return self.discriminator(batch)

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

    def batch_generate(self, cls_idx, total, epochs, show_progress=True):
        pbar = tqdm(total=total)
        count = 0
        while count < total:
            self.init()
            if self.train(epochs, show_progress=show_progress):
                self.save_sampler(cls_idx)
                count += 1
                pbar.update(1)

    def get_training_success_rate(self, total, epochs, show_progress=False):
        iters = []
        for _ in (bar := trange(total)):
            self.init()
            if self.train(epochs, show_progress=show_progress):
                iters.append(self.iteration)
            bar.set_postfix({'count': len(iters)})
        return iters
