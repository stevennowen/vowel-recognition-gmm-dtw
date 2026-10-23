import numpy as np


def dtw(template, inp):
    cost = np.linalg.norm(template[:, None, :] - inp[None, :, :], axis=2)
    n, m = cost.shape

    total = np.full((n, m), np.inf)
    step = np.zeros((n, m), dtype=int)
    total[0, 0] = cost[0, 0]

    for j in range(1, m):
        prev = total[:, j - 1]
        stay = prev
        diagonal = np.r_[np.inf, prev[:-1]]
        skip = np.r_[np.inf, np.inf, prev[:-2]][:n]
        options = np.vstack([stay, diagonal, skip])

        step[:, j] = options.argmin(axis=0)
        total[:, j] = options.min(axis=0) + cost[:, j]

    path = [n - 1]
    for j in range(m - 1, 0, -1):
        i = path[-1]
        path.append(i - step[i, j])
    path.reverse()

    return total[-1, -1], np.array(path)


def average_template(sequences):
    lengths = [len(seq) for seq in sequences]
    reference_index = np.argsort(lengths)[len(lengths) // 2]
    reference = sequences[reference_index]

    aligned = [reference]
    for index, seq in enumerate(sequences):
        if index == reference_index:
            continue
        cost, path = dtw(seq, reference)
        if np.isfinite(cost):
            aligned.append(seq[path])

    return np.mean(aligned, axis=0)
