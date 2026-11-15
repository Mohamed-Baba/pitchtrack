"""Optimal assignment between tracks and detections.

The tracker never matches greedily: at each frame it solves a linear assignment problem
whose cost is 1 - IoU (or a Mahalanobis distance), with a gate that forbids implausible
pairs. A pair can be left unmatched at a price. "Solve, then drop the pairs above the gate"
is not the optimal rule (it maximises the number of matches first and can break an obvious
pair): the augmented matrix below gives the optimum with the gate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6   # cost of a forbidden pair (any finite number much larger than the real costs)


# --------------------------------------------------------------------------- assignment with a gate
def linear_assignment(cost, max_cost: float):
    """Optimal partial matching between the rows and the columns of `cost` (n x m).

    Minimise  sum of cost[i, j] over matched pairs  +  max_cost / 2 per unmatched row or column.
    A pair with cost > max_cost is therefore never worth matching. Solve it exactly with one
    call to scipy's linear_sum_assignment on the (n + m) x (n + m) augmented matrix

        [ C_gated            diag(max_cost/2) ]      C_gated: cost, with BIG above max_cost
        [ diag(max_cost/2)   0                ]      off-diagonal entries of the diag blocks: BIG

    Returns (matches (K, 2) int array of (row, col), unmatched_rows, unmatched_cols).
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=np.int64), np.arange(n), np.arange(m)
    
    # 1. Interdire les associations réelles dont le coût dépasse max_cost.
    C_gated = np.where(cost > max_cost, BIG, cost)

    # 2. Bloc supérieur droit : non-association des lignes réelles.
    top_right = np.full((n, n), BIG, dtype=np.float64)
    np.fill_diagonal(top_right, max_cost / 2)

    # 3. Bloc inférieur gauche : non-association des colonnes réelles.
    bottom_left = np.full((m, m), BIG, dtype=np.float64)
    np.fill_diagonal(bottom_left, max_cost / 2)

    # 4. Bloc inférieur droit : associations fictives gratuites.
    bottom_right = np.zeros((m, n), dtype=np.float64)

    # 5. Construire la matrice augmentée (n+m) x (n+m).
    C_aug = np.block([
        [C_gated, top_right],
        [bottom_left, bottom_right],
    ])

    # 6. Résoudre le problème d'affectation complet.
    rows, cols = linear_sum_assignment(C_aug)

    # 7. Garder uniquement les associations réelles.
    real = (rows < n) & (cols < m)
    matches = np.column_stack((rows[real], cols[real])).astype(np.int64)

    # 8. Déterminer les lignes et colonnes non associées.
    matched_rows = matches[:, 0]
    matched_cols = matches[:, 1]

    unmatched_rows = np.setdiff1d(np.arange(n), matched_rows)
    unmatched_cols = np.setdiff1d(np.arange(m), matched_cols)

    return matches, unmatched_rows, unmatched_cols



# --------------------------------------------------------------------------- Hungarian method
def hungarian(cost):
    """Minimum-cost assignment written from scratch (Hungarian method with potentials).

    Every row of an n x m matrix with n <= m gets a distinct column (the matrix is transposed
    when n > m). Same output as scipy.optimize.linear_sum_assignment: (rows, cols) sorted by
    row. Shortest augmenting paths with dual potentials u (rows) and v (columns), O(n^2 m):
    reduced costs cost[i, j] - u[i] - v[j] stay >= 0 and are 0 on the matched pairs.
    """

    C = np.asarray(cost, dtype=np.float64)

    # 1-based arrays u (n + 1), v (m + 1), p (m + 1: row matched to each column,
    # p[0] = row being inserted) and way (m + 1). For each row i: grow a shortest-path tree
    # over the columns (minv, used), shift the potentials by the smallest reduced cost delta,
    # stop at a free column, then flip the path stored in `way`.
    
    # 1. Transposer si nécessaire pour avoir n <= m.
    transposed = C.shape[0] > C.shape[1]

    if transposed:
        C = C.T

    n, m = C.shape

    if n == 0:
        return np.zeros(0, dtype=np.int64),np.zeros(0, dtype=np.int64)
        
    # 2. Potentiels et tableaux auxiliaires (indexation à partir de 1).
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    # 3. Insérer les lignes une par une.
    for i in range(1, n + 1):

        p[0] = i
        j0 = 0

        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)

        # 4. Chercher un chemin augmentant de coût minimal.
        while True:

            used[j0] = True
            i0 = p[j0]

            delta = np.inf
            j1 = 0

            for j in range(1, m + 1):

                if not used[j]:

                    # Coût réduit.
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]

                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0

                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # 5. Mettre à jour les potentiels.
            for j in range(m + 1):

                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            # On a atteint une colonne libre.
            if p[j0] == 0:
                break

        # 6. Reconstruire le chemin augmentant.
        while True:

            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

            if j0 == 0:
                break

    # 7. Extraire les associations.
    cols = np.flatnonzero(p[1:] > 0)
    rows = p[1:][cols] - 1

    # cols est déjà indexé à partir de zéro.
    # rows contient les lignes correspondantes.
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]

    # 8. Annuler la transposition éventuelle.
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows = rows[order]
        cols = cols[order]

    return rows.astype(np.int64), cols.astype(np.int64)

