use ndarray::{Array, Array2, Axis};
use rayon::prelude::*;
use rand::prelude::*;

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
    let mut sum = 0.0;
    let len = u.len();

    for i in 0..len {
        let diff = u[i] - v[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

pub fn sample_rows<T>(arr: &Array2<T>, n: usize) -> Array2<T>
where
    T: Clone,
{
    // Get the number of rows in the array
    let num_rows = arr.nrows();

    // Generate a random slice of indices, without replacement
    let mut rng = rand::thread_rng();
    let selected_indices: Vec<usize> = (0..num_rows).choose_multiple(&mut rng, n);

    // Select rows based on the chosen indices
    arr.select(Axis(0), &selected_indices)
}