from sklearn_numba_dpex.kmeans.kernels import (
    make_lloyd_single_step_fixed_window_kernel,
)
import dpctl.tensor as dpt
import numpy as np
import timeit
from sklearn_numba_dpex.common._utils import _get_global_mem_cache_size

n_iter = 100

n_samples = 5000000
n_features = 14
n_clusters = 127
return_assignments = True
check_strict_convergence = False
dtype = np.float32

X_t = dpt.empty((n_features, n_samples), dtype=dtype)

compute_dtype = X_t.dtype.type

device = X_t.device.sycl_device
max_work_group_size = device.max_work_group_size
sub_group_size = min(device.sub_group_sizes)
global_mem_cache_size = _get_global_mem_cache_size(device)

centroids_t = dpt.empty((n_features, n_clusters), dtype=dtype)
sample_weight = dpt.empty((n_samples,), dtype=dtype)
centroids_half_l2_norm = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
new_assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)
assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)


# allocation of one scalar where we store the result of strict convergence check
strict_convergence_status = dpt.empty(1, dtype=np.uint32, device=device)


(
    n_centroids_private_copies,
    fused_lloyd_fixed_window_single_step_kernel,
) = make_lloyd_single_step_fixed_window_kernel(
    n_samples,
    n_features,
    n_clusters,
    return_assignments,
    check_strict_convergence,
    sub_group_size,
    global_mem_cache_size,
    centroids_private_copies_max_cache_occupancy=0.7,
    work_group_size="max",
    dtype=dtype,
    device=device,
)

print(n_centroids_private_copies)

new_centroids_t_private_copies = dpt.empty(
    (n_centroids_private_copies, n_features, n_clusters),
    dtype=compute_dtype,
    device=device,
)
cluster_sizes_private_copies = dpt.empty(
    (n_centroids_private_copies, n_clusters),
    dtype=compute_dtype,
    device=device,
)

print("Compiling...")
fused_lloyd_fixed_window_single_step_kernel(
    X_t,
    sample_weight,
    centroids_t,
    centroids_half_l2_norm,
    assignments_idx,
    # OUT:
    new_assignments_idx,
    strict_convergence_status,
    new_centroids_t_private_copies,
    cluster_sizes_private_copies,
)
print("Compiling...OK")

print(f"n_samples={n_samples} n_features={n_features} n_clusters={n_clusters}")

print("Running...")
t0 = timeit.default_timer()
for i in range(n_iter):
    fused_lloyd_fixed_window_single_step_kernel(
        X_t,
        sample_weight,
        centroids_t,
        centroids_half_l2_norm,
        assignments_idx,
        # OUT:
        new_assignments_idx,
        strict_convergence_status,
        new_centroids_t_private_copies,
        cluster_sizes_private_copies,
    )
t1 = timeit.default_timer()
print(f"Running...OK. Time: {t1-t0}")
