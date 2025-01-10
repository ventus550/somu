// fn norm(v: &[f64]) -> f64 {
    //     v.iter().map(|&x| x * x).sum::<f64>().sqrt()
// }

mod distance;

pub use crate::distance::compute_distance_matrix;
pub use crate::distance::cdist_seuclidean;

use itertools::Itertools;
use pyo3::prelude::*;
use ndarray::{Array1, Array2, Axis, s};
use numpy::{IntoPyArray, PyArray1};
use ndarray_stats::QuantileExt;
use std::f64::consts::E;
use rand::prelude::*;
// use rayon::prelude::*;

fn sample_rows<T>(arr: &Array2<T>, n: usize) -> Array2<T> where T: Clone{
    // Get the number of rows in the array
    let num_rows = arr.nrows();
    
    // Generate a random slice of indices, without replacement
    let mut rng = rand::thread_rng();
    let selected_indices: Vec<usize> = (0..num_rows).choose_multiple(&mut rng, n);

    // Select rows based on the chosen indices
    arr.select(Axis(0), &selected_indices).to_owned()
}

/// Self-Organizing Map (SOM) implementation.
fn som(
    x: &Array2<f64>,
    grid: &Array2<f64>,
    epochs: usize,
    sigma_initial: f64,
) -> Array2<f64> {
    let (n, d) = x.dim();
    let (m, _) = grid.dim();

    // Initialize weights randomly from input data
    let mut centroids = sample_rows(x, m);

    // Precompute grid distances
    let grid_distances = compute_distance_matrix(grid, grid);
    
    // Training loop
    for e in 0..epochs {
        let sigma = sigma_initial * E.powf(-(e as f64) / (epochs as f64));
        
        // Compute distances between inputs and centroids once per epoch
        let dst = compute_distance_matrix(x, &centroids);

        // Find best matching units (BMUs) for each input
        let bmu: Vec<usize> = dst.outer_iter()
        .map(|row| row.argmin().unwrap())
        .collect();
    
        // Compute neighborhood influence for all neurons (a function of distance)
        let influence = grid_distances.mapv(|dist| (-dist.powi(2) / (2.0 * sigma.powi(2)))).exp();

        // Update weights
        let mut numerator = Array2::<f64>::zeros((m, d));
        let mut denominator = Array1::<f64>::zeros(m);

        // Loop through the dataset and update weights
        for i in 0..n {
            // Compute updates
            let i_slice = &influence.row(bmu[i]).insert_axis(Axis(1));
            let x_slice = &x.row(i).insert_axis(Axis(0));
            let update = i_slice *  x_slice; // Shape (m, d)

            numerator += &update;
            denominator += i_slice;
        }

        // centroids = &numerator / &denominator.insert_axis(Axis(1)) crashes kernel?
        for k in 0..m {
            for j in 0..d {
                centroids[[k, j]] = numerator[[k, j]] / denominator[k];
            }
        }
    }

    centroids
}





#[pyfunction]
fn distance_matrix(p: Vec<Vec<f64>>, q: Vec<Vec<f64>>, py: Python) ->  PyResult<Bound<PyArray1<f64>>> {
    let num_rows_p = p.len();
    let num_cols = p[0].len();
    let num_rows_q = q.len();

    // Flatten p and q for the calculation
    let p_flat: Vec<f64> = p.iter().flat_map(|v| v.clone()).collect();
    let q_flat: Vec<f64> = q.iter().flat_map(|v| v.clone()).collect();

    let mut dm = vec![0.0; num_rows_p * num_rows_q];
    cdist_seuclidean(&p_flat, &q_flat, &mut dm, num_rows_q, num_cols);

    // let dm_2d = ndarray::Array::from_shape_vec((num_rows_p, num_rows_q), dm);
    // Convert the flat distance matrix into a 2D vector for easier Python use
    // vec![vec![0.0]]
    // let mut result = vec![vec![0.0; num_rows_q]; num_rows_p];
    // for i in 0..num_rows_p {
    //     for j in 0..num_rows_q {
        //         result[i][j] = dm[i * num_rows_q + j];
        //     }
    // }

    Ok(dm.into_pyarray(py))
}


#[pyfunction]
fn identity<'py>(py: Python<'py>, arr: Vec::<f32>) -> PyResult<Bound<PyArray1<f32>>> {
    let len = arr.len();
    let mut arr2: Vec<f32> = vec![0.0; len];
    for i in 0..len {
        arr2[i] = arr[i];
    }
    Ok(arr2.into_pyarray(py))
}

/// Python wrapper for the `som` function
#[pyfunction]
fn som_wrapper(
    // py: Python<'py>,
    x: Vec<Vec<f64>>,
    grid: Vec<Vec<f64>>,
    epochs: usize,
    sigma_initial: f64,
) -> Vec<Vec<f64>> {
    let x_array = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
    let grid_array = Array2::from_shape_vec((grid.len(), grid[0].len()), grid.into_iter().flatten().collect()).unwrap();
    let result = som(&x_array, &grid_array, epochs, sigma_initial);
    // PyArray2::from_array(py, &result)
    result
    .axis_iter(Axis(0)) // Iterate over rows (each row is a 1D array)
    .map(|row| row.to_vec()) // Convert each row to Vec<f64>
    .collect::<Vec<_>>() // Collect the Vec<f64> into a Vec
}


#[pyfunction]
/// Creates a grid
fn generate_grid(dims: Vec<usize>) -> Vec<Vec<usize>> {
    dims.into_iter()
        .map(|dim_size| 0..dim_size)
        .multi_cartesian_product()
        .collect()
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_grid, m)?)?;
    m.add_function(wrap_pyfunction!(som_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    Ok(())
}
