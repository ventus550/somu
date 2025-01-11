use ndarray::{Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use rayon::prelude::*;

use crate::linalg::{compute_distance_matrix, sample_rows};
use std::f64::consts::E;

pub fn organize(
    x: &Array2<f64>,
    unit_dst: &Array2<f64>,
    dst: &Array2<f64>,
    m: usize,
    sigma: f64,
) -> Array2<f64> {
    let (n, d) = x.dim();

    // Find best matching units (BMUs) for each input
    let bmu: Vec<usize> = dst.outer_iter().map(|row| row.argmin().unwrap()).collect();

    // Compute neighborhood influence for all neurons (a function of distance)
    let influence = unit_dst
        .mapv(|dist| (-dist.powi(2) / (2.0 * sigma.powi(2))))
        .exp();

    // Parallel computation for the batch updates
    let (numerator, denominator) = (0..n)
        .into_par_iter()
        .map(|i| {
            let influence_row = influence.row(bmu[i]);
            let update = &influence_row.insert_axis(Axis(1)) * &x.row(i); // Shape (m, d)
            (update, influence_row.to_owned())
        })
        .reduce(
            || (Array2::zeros((m, d)), Array1::zeros(m)),
            |(num1, den1), (num2, den2)| (num1 + num2, den1 + den2),
        );

    // Update centroid positions
    &numerator / &(denominator + f64::EPSILON).insert_axis(Axis(1))
}

/// Self-Organizing Map (SOM) implementation.
pub fn som_functional(x: &Array2<f64>, units: &Array2<f64>, epochs: usize, sigma_initial: f64) -> Array2<f64> {
	let m = units.nrows();

    // Initialize weights randomly from the input data
    let mut centroids = sample_rows(x, m);

    // Precompute distances between units
    let unit_dst = compute_distance_matrix(units, units);

    // Training loop
    for e in 0..epochs {
        let sigma = sigma_initial * E.powf(-(e as f64) / (epochs as f64));
		let dst = compute_distance_matrix(&x, &centroids);
		centroids = organize(&x, &unit_dst, &dst, m, sigma)
    }
    centroids
}