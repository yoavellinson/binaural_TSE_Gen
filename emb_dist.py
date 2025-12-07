import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def load_embeddings_from_dir(dir_path):
    embeddings = []
    names = []

    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".pt"):
            emb = torch.load(os.path.join(dir_path, fname))
            embeddings.append(emb.squeeze())
            names.append(fname)
    
    # stack to [N, 1029, 64]
    E = torch.stack(embeddings)
    return E, names

E, names = load_embeddings_from_dir("/home/workspace/yoavellinson/binaural_TSE_Gen/ae_res")
print(E.shape)  # e.g. [N, 1029, 64]
E_flat = E.view(E.size(0), -1)   # [N, 65856]



def gaussian_affinity(E):
    # Flatten embeddings
    E_flat = E.view(E.size(0), -1)

    # Pairwise distances
    dists = torch.cdist(E_flat, E_flat)
    # dists = dists / dists.max()
    sigma = 0.05 
    print(sigma)
    K = torch.exp(- (dists ** 2) / (2 * sigma ** 2))
    return K

def plot_affinity_matrix(affinity, names):
    A = affinity.cpu().numpy()
    N = len(names)

    plt.figure(figsize=(12, 10))
    plt.imshow(A, cmap='viridis')

    plt.colorbar(label="Cosine Similarity")

    # Set ticks
    plt.xticks(ticks=np.arange(N), labels=names, rotation=90, fontsize=7)
    plt.yticks(ticks=np.arange(N), labels=names, fontsize=7)

    plt.title("Embedding Affinity Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig('aff.png')
# compute pairwise cosine similarity
affinity = gaussian_affinity(E_flat)
print(affinity.shape)  # [N, N]
plot_affinity_matrix(affinity, names)