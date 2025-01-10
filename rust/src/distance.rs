use rayon::prelude::*;
use ndarray::{Array, Array2};

/// Computes the pairwise distance matrix between two sets of points p and q using broadcasting.
pub fn compute_distance_matrix(p: &Array2<f64>, q: &Array2<f64>) -> Array2<f64> {
    let (n_p, d_p) = p.dim();
    let (n_q, d_q) = q.dim();

    assert_eq!(d_p, d_q, "Points in p and q must have the same dimension.");


    // Flatten the input arrays into vectors for cdist_seuclidean
    let p_flat: Vec<f64> = p.iter().cloned().collect();
    let q_flat: Vec<f64> = q.iter().cloned().collect();

    // Allocate space for the resulting flat distance matrix
    let mut dm_flat = vec![0.0; n_p * n_q];

    // Call cdist_seuclidean to compute distances
    cdist_seuclidean(&p_flat, &q_flat, &mut dm_flat, n_q, d_p);

    // Convert the flat distance matrix into a 2D Array2
    Array::from_shape_vec((n_p, n_q), dm_flat)
        .expect("Failed to reshape the flat distance matrix into 2D")
}

#[inline(always)]
pub fn cdist_seuclidean(
    xa: &[f64],
    xb: &[f64],
    dm: &mut [f64],
    num_rows_b: usize,
    num_cols: usize,
) {
    // Use parallel iterators to compute distances
    dm.par_chunks_mut(num_rows_b)
        .enumerate()
        .for_each(|(i, dm_chunk)| {
            let u_start = i * num_cols;
            let u = &xa[u_start..u_start + num_cols];

            dm_chunk.iter_mut().enumerate().for_each(|(j, dm_val)| {
                let v_start = j * num_cols;
                let v = &xb[v_start..v_start + num_cols];
                *dm_val = seuclidean_distance(u, v);
            });
        });
}

#[inline(always)]
fn seuclidean_distance(u: &[f64], v: &[f64]) -> f64 {
    // We compute the sum of squared differences
    u.iter()
        .zip(v.iter())
        .map(|(u_val, v_val)| {
            let d = u_val - v_val;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}
