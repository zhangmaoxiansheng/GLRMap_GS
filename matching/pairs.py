"""Current sparse-covisibility graph: connectivity, neighbor support, then score fill."""
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def sparse_pairs(model, images):
    reliable = {pid for pid, point in model.points3D.items()
                if point.error <= 1 and point.track.length() >= 3}
    # GLOMAP can attach multiple 2D observations to one 3D ID in an image.
    # Such duplicates must not inflate pair scores or create false overlap.
    observed = [np.array(sorted({p.point3D_id for p in image.points2D
                              if p.has_point3D() and p.point3D_id in reliable}), np.int64)
                for image in images]
    if len(images) < 2:
        return {}
    point_ids = np.array(sorted(reliable), np.int64)
    rows = np.repeat(np.arange(len(images)), [len(ids) for ids in observed])
    columns = np.searchsorted(point_ids, np.concatenate(observed))
    visibility = csr_matrix((np.ones(len(rows), np.int32), (rows, columns)),
                            shape=(len(images), len(point_ids)))
    # Integer incidence products count shared points; bool multiplication would
    # retain only connectivity. At 1000 views the dense result is just 4 MB.
    counts = (visibility @ visibility.T).toarray()
    return {(i, j): int(counts[i, j])
            for i, j in itertools.combinations(range(len(images)), 2)}


def select_pairs(model, images, initial_budget, budget, min_neighbors=3):
    """Return name pairs and scores; equivalent to current MST→1800→3600 selection.

    Images must use the name-sorted order returned by sfm.geometry.model_data.
    No temporal/spatial hard cap, voxel prior, or angle-based pair rejection.
    """
    names = [image.name for image in images]
    scores = sparse_pairs(model, images)
    if not len(names)-1 <= initial_budget < budget <= len(scores):
        raise ValueError('Require n-1 <= initial_budget < budget <= unique image pairs')
    weights = np.zeros((len(names), len(names)))
    for (a, b), value in scores.items():
        if value > 0:
            weights[a, b] = weights[b, a] = -value
    tree = minimum_spanning_tree(weights).tocoo()
    chosen = {tuple(sorted((int(a), int(b)))) for a, b in zip(tree.row, tree.col)}
    if len(chosen) != len(names)-1:
        raise ValueError('Reliable sparse co-visibility graph is disconnected')
    for pair in sorted(scores, key=lambda p: (scores[p], -p[0], -p[1]), reverse=True):
        if len(chosen) >= initial_budget:
            break
        if scores[pair] > 0:
            chosen.add(pair)
    if len(chosen) != initial_budget:
        raise ValueError('Insufficient positive co-visibility pairs for initial budget')
    prior = chosen.copy()
    degree = np.zeros(len(names), int)
    for a, b in chosen:
        degree[a] += 1
        degree[b] += 1
    nominations = []
    for i in sorted(range(len(names)), key=lambda i: (degree[i], i)):
        while degree[i] < min_neighbors and len(chosen) < budget:
            options = [j for j in range(len(names)) if j != i
                       and tuple(sorted((i, j))) not in chosen
                       and scores[(min(i, j), max(i, j))] >= 30]
            if not options:
                break
            medium = [j for j in options if 2 <= abs(j-i) <= 8]
            j = max(medium or options, key=lambda j: (
                scores[(min(i, j), max(i, j))], -degree[j], -abs(j-i), -j))
            pair = tuple(sorted((i, j)))
            chosen.add(pair)
            degree[i] += 1
            degree[j] += 1
            nominations.append([names[pair[0]], names[pair[1]]])
    ranking = sorted(scores, key=lambda ij: (
        scores[ij] > 0, scores[ij], -ij[0], -ij[1]), reverse=True)
    for pair in ranking:
        if len(chosen) == budget:
            break
        if pair not in chosen and scores[pair] > 0:
            chosen.add(pair)
    if len(chosen) != budget:
        raise ValueError('Insufficient positive co-visibility pairs for final budget')
    pairs = [[names[i], names[j]] for i, j in sorted(chosen)]
    degree[:] = 0
    for i, j in chosen:
        degree[i] += 1
        degree[j] += 1
    return pairs, dict(
        initial_pairs=len(prior), selected_pairs=len(pairs), min_neighbors_target=min_neighbors,
        balanced_nominations=nominations, per_image_neighbors=degree.tolist(),
        selected_scores=[dict(pair=[names[i], names[j]], shared_points=scores[(i, j)])
                         for i, j in sorted(chosen)],
        reliable_points='point error <=1 px and sparse track length >=3',
        selection='Maximum covisibility spanning tree; initial score fill; underserved '
                  'views prefer gaps 2..8 with >=30 shared points; final score fill.')
