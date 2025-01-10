mod linalg;

use itertools::Itertools;
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::E;

pub use crate::linalg::{cdist_seuclidean, compute_distance_matrix, sample_rows};

/// Self-Organizing Map (SOM) implementation.
fn som(x: &Array2<f64>, units: &Array2<f64>, epochs: usize, sigma_initial: f64) -> Array2<f64> {
    let (n, d) = x.dim();
    let (m, _) = units.dim();

    // Initialize weights randomly from the input data
    let mut centroids = sample_rows(x, m);

    // Precompute distances between units
    let unit_dst = compute_distance_matrix(units, units);

    // Training loop
    for e in 0..epochs {
        let sigma = sigma_initial * E.powf(-(e as f64) / (epochs as f64));

        // Compute distances between inputs and centroids once per epoch
        let dst = compute_distance_matrix(x, &centroids);

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
        centroids = &numerator / &(denominator + f64::EPSILON).insert_axis(Axis(1));
    }

    centroids
}

#[pyfunction]
fn distance_matrix(
    p: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    py: Python,
) -> PyResult<Bound<PyArray1<f64>>> {
    let num_rows_p = p.len();
    let num_cols = p[0].len();
    let num_rows_q = q.len();

    // Flatten p and q for the calculation
    let p_flat: Vec<f64> = p.iter().flat_map(|v| v.clone()).collect();
    let q_flat: Vec<f64> = q.iter().flat_map(|v| v.clone()).collect();

    let mut dm = vec![0.0; num_rows_p * num_rows_q];
    cdist_seuclidean(&p_flat, &q_flat, &mut dm, num_rows_q, num_cols);
    Ok(dm.into_pyarray(py))
}

/// Python wrapper for the `som` function
#[pyfunction]
fn som_wrapper(
    // py: Python<'py>,
    x: Vec<Vec<f64>>,
    units: Vec<Vec<f64>>,
    epochs: usize,
    sigma_initial: f64,
) -> Vec<Vec<f64>> {
    let x_array =
        Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
    let units_array = Array2::from_shape_vec(
        (units.len(), units[0].len()),
        units.into_iter().flatten().collect(),
    )
    .unwrap();
    let result = som(&x_array, &units_array, epochs, sigma_initial);
    // PyArray2::from_array(py, &result)
    result
        .axis_iter(Axis(0)) // Iterate over rows (each row is a 1D array)
        .map(|row| row.to_vec()) // Convert each row to Vec<f64>
        .collect::<Vec<_>>() // Collect the Vec<f64> into a Vec
}

#[pyfunction]
/// Creates a regular hyperdimensional grid
fn generate_grid(dims: Vec<usize>) -> Vec<Vec<usize>> {
    dims.into_iter()
        .map(|dim_size| 0..dim_size)
        .multi_cartesian_product()
        .collect()
}

// #[pyclass]
// struct SOM {
//     units: Array2<f64>,
// }

// #[pymethods]
// impl SOM {
//     #[new]
//     fn new(units: Vec<Vec<f64>>) -> Self {
//         let units = Array2::from_shape_vec((units.len(), units[0].len()), units.into_iter().flatten().collect()).unwrap();
//         SOM{units}
//     }

//     fn units(&self) -> PyResult<>

    
// }

#[pymodule]
mod rust {
    use super::*;

    #[pymodule_export]
    use super::som_wrapper;

    #[pymodule_export]
    use super::distance_matrix;

    #[pymodule_export]
    use super::generate_grid;

    // #[pymodule_export]
    // use super::SOM;

    #[pyfunction]
    fn identity<'py>(py: Python<'py>, arr: Vec<f32>) -> PyResult<Bound<PyArray1<f32>>> {
        let len = arr.len();
        let mut arr2: Vec<f32> = vec![0.0; len];
        for i in 0..len {
            arr2[i] = arr[i];
        }
        Ok(arr2.into_pyarray(py))
    }
}