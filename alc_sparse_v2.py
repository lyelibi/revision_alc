import numpy as np
from numba import vectorize
from sklearn.preprocessing import StandardScaler
from umap_contructor import build_graph

@vectorize('float32(float32, int32)', nopython=True)
def lc(corr, size):
    if size <=1.0:
        return 0.0
    if corr <= size:
        return 0.0
    if size**2 - corr <= 0.0:
        corr-=1e-3
    return 0.5*( np.log(size/corr) +  (size - 1.0)*np.log( (size**2 - size) / ( size**2 - corr) )  )

@vectorize('float32(float32, float32, float32, int32)', nopython=True)
def merge(corr_com, corr_neighbors, cross_neighbors , size):
    corr = corr_com + corr_neighbors + 2.0 * cross_neighbors
    if corr <= 0:
        return 0.0
    if size <=1.0:
        return 0.0
    if corr <= size:
        return 0.0
    if size**2 - corr <= 0.0:
        corr-=1e-3
    return 0.5*( np.log(size/corr) +  (size - 1.0)*np.log( (size**2 - size) / ( size**2 - corr) )  )

def alc_sparse_v2(data, n_neighbors=100):
    data = StandardScaler().fit_transform(data.T).T
    G = build_graph(data, metric='correlation', n_neighbors = n_neighbors, verbose=True).tolil()
    N = G.shape[0]
    communities = np.arange(N, dtype=np.int32)
    tracker = np.arange(N, dtype=np.int32)
    ns = np.ones(N, dtype=np.int32)
    all_seen = False    
    while not all_seen:
        all_seen = True
        while len(tracker) > 1:
            community = np.random.choice(tracker)
            community_neighbors = G[community].nonzero()[1]  # Get the indices of nonzero entries
            community_neighbors = community_neighbors[community_neighbors != community]
            A = lc(G[community, community], ns[community])
            B = lc(G[community_neighbors, community_neighbors].toarray(), ns[community_neighbors])
            C = merge(G[community, community],
                      G[community_neighbors, community_neighbors].toarray(),
                      G[community, community_neighbors].toarray(),
                      ns[community] + ns[community_neighbors])
            scores = C - (A + B)
            if np.all( scores <= 0 ):
                tracker = np.setdiff1d(tracker,community)
                continue
            pick = community_neighbors[np.argmax(scores)]  
            pick_neighbors = G[pick].nonzero()[1]
            pick_neighbors = pick_neighbors[pick_neighbors != community]
            pick_neighbors = pick_neighbors[pick_neighbors != pick]
            G[community, pick_neighbors] += G[pick, pick_neighbors]
            G[pick_neighbors, community] = G[community, pick_neighbors]
            G[community, community] += G[pick, pick] + 2 * G[pick, community]
            G[pick_neighbors, pick] = 0
            G[community, pick] = 0
            communities[communities == pick] = community
            ns[community] += ns[pick]
            ns[pick] = 0
            all_seen = False
            tracker = np.unique(communities)
        if all_seen:
            return communities
