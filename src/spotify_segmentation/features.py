from sklearn.decomposition import PCA

def make_pca(n_components: int):
    if not n_components or n_components <= 0:
        return None
    return PCA(n_components=n_components, random_state=0)
