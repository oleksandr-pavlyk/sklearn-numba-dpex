import math
from functools import lru_cache

import numba_dpex as dpex


# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_compute_inertia_kernel(
    n_samples,
    n_features,
    work_group_size,
    dtype,
):

    zero = dtype(0.0)

    @dpex.kernel
    # fmt: off
    def compute_inertia(
        X_t,                          # IN READ-ONLY   (n_features, n_samples)
        sample_weight,                # IN READ-ONLY   (n_features,)
        centroids_t,                  # IN READ-ONLY   (n_features, n_clusters)
        assignments_idx,              # IN READ-ONLY   (n_samples,)
        per_sample_inertia,           # OUT            (n_samples,)
    ):
    # fmt: on

        sample_idx = dpex.get_global_id(0)

        if sample_idx >= n_samples:
            return

        inertia = zero

        centroid_idx = assignments_idx[sample_idx]

        for feature_idx in range(n_features):

            diff = X_t[feature_idx, sample_idx] - centroids_t[feature_idx, centroid_idx]
            inertia += diff * diff

        per_sample_inertia[sample_idx] = inertia * sample_weight[sample_idx]

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return compute_inertia[global_size, work_group_size]
