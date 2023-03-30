from kmeans_dpcpp import fused_lloyd_single_step
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

n_centroids_private_copies = 1024

X_t = dpt.empty((n_features, n_samples), dtype=dtype)

compute_dtype = X_t.dtype.type

device = X_t.device.sycl_device
max_work_group_size = device.max_work_group_size
sub_group_size = min(device.sub_group_sizes)
global_mem_cache_size = _get_global_mem_cache_size(device)

centroids_t = dpt.empty((n_features, n_clusters), dtype=dtype)
sample_weight = dpt.empty((n_samples,), dtype=dtype)
centroids_half_l2_norm = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
assignments_idx = dpt.empty(n_samples, dtype=np.int32, device=device)


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

q = X_t.sycl_queue
ht, _ = fused_lloyd_single_step(
    X_t, sample_weight, centroids_t, centroids_half_l2_norm, assignments_idx,
    new_centroids_t_private_copies,
    cluster_sizes_private_copies,
    8,      # centroids_window_height
    64,    # work_group_size
    q       # sycl_queue
)
ht.wait()

print("Compiling...OK")
print(f"n_samples={n_samples} n_features={n_features} n_clusters={n_clusters}")

print("Running...")
t0 = timeit.default_timer()
for i in range(n_iter):
    ht, _ = fused_lloyd_single_step(
       X_t, sample_weight, centroids_t, centroids_half_l2_norm, assignments_idx,
       new_centroids_t_private_copies,
       cluster_sizes_private_copies,
       8,      # centroids_window_height
       64,     # work_group_size
       q       # sycl_queue
    )
    ht.wait()
t1 = timeit.default_timer()
print(f"Running...OK. Time: {t1-t0}")


