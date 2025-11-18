import random
from torch.utils.data import Sampler


class EpisodicSampler(Sampler):
    """
    Creates episodes for few-shot learning:
        - N-way: number of classes per episode
        - K-shot: number of support samples per class
        - Q-query: number of query samples per class
        - episodes_per_epoch: number of episodes per epoch
    """

    def __init__(self, labels, n_way, k_shot, q_query, episodes_per_epoch):
        super().__init__(None)
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch

        # Build dictionary: class → list of sample indices
        self.class_to_indices = {}
        for idx, lbl in enumerate(labels):
            if lbl not in self.class_to_indices:
                self.class_to_indices[lbl] = []
            self.class_to_indices[lbl].append(idx)

        # Only keep classes with enough samples
        for cls in list(self.class_to_indices.keys()):
            if len(self.class_to_indices[cls]) < (k_shot + q_query):
                raise ValueError(
                    f"Class {cls} does not have enough samples "
                    f"for {k_shot}-shot and {q_query}-query episodes."
                )

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # Sample N classes
            episode_classes = random.sample(
                list(self.class_to_indices.keys()),
                self.n_way
            )

            episode_indices = []

            for cls in episode_classes:
                indices = self.class_to_indices[cls]
                # Randomly pick K + Q samples
                chosen = random.sample(indices, self.k_shot + self.q_query)
                episode_indices.extend(chosen)

            yield episode_indices
