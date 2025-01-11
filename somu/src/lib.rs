mod algo;
mod linalg;
mod utils;

use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use std::f64::consts::E;

use crate::algo::{organize, som_functional};
use crate::linalg::{cdist_seuclidean, compute_distance_matrix, sample_rows};
use crate::utils::{array2, generate_grid};

#[pyfunction]
fn distance_matrix(p: Vec<Vec<f64>>, q: Vec<Vec<f64>>, py: Python) -> Bound<PyArray1<f64>> {
    let num_rows_p = p.len();
    let num_cols = p[0].len();
    let num_rows_q = q.len();

    // Flatten p and q for the calculation
    let p_flat: Vec<f64> = p.iter().flat_map(|v| v.clone()).collect();
    let q_flat: Vec<f64> = q.iter().flat_map(|v| v.clone()).collect();

    let mut dm = vec![0.0; num_rows_p * num_rows_q];
    cdist_seuclidean(&p_flat, &q_flat, &mut dm, num_rows_q, num_cols);
    dm.into_pyarray(py)
}

/// Functional self-organizing map
#[pyfunction]
fn som(
    x: Vec<Vec<f64>>,
    units: Vec<Vec<f64>>,
    epochs: usize,
    sigma_initial: f64,
    py: Python,
) -> Bound<PyArray2<f64>> {
    som_functional(&array2(x), &array2(units), epochs, sigma_initial).into_pyarray(py)
}
#[pyclass]
#[derive(Clone)]
/// Self-Organizing Map (SOM) class.
/// 
/// A Self-Organizing Map (SOM) is an unsupervised learning algorithm that maps high-dimensional data to a lower-dimensional grid, preserving topological properties. This implementation includes the ability to train the SOM on input data and obtain a grid of units and quantization errors.
struct SOM {
    /// Distance matrix between the most recent input data and the current centroids.
    dst: Array2<f64>,
    /// The grid of units in the SOM.
    units: Array2<f64>,
    /// Centroids (weights) for each unit on the grid.
    centroids: Option<Array2<f64>>,
}

#[pymethods]
impl SOM {
    /// Creates a new instance of the SOM with specified unit grid dimensions.
    /// 
    /// # Arguments
    /// * `dims` - A vector specifying the dimensions of the grid (rows and columns).
    ///
    /// # Returns
    /// A new instance of the SOM class.
    #[new]
    fn new(dims: Vec<usize>) -> Self {
        SOM {
            dst: array2(vec![vec![-1.0]]),
            units: array2(generate_grid(dims)),
            centroids: None,
        }
    }

    /// Getter for the `units` field.
    /// 
    /// # Arguments
    /// * `py` - Python interpreter context.
    ///
    /// # Returns
    /// A Python array representing the units (grid) of the SOM.
    #[getter]
    fn units(&self, py: Python) -> Py<PyArray2<f64>> {
        self.units.to_pyarray(py).into()
    }

    /// Getter for the `centroids` field.
    /// 
    /// # Arguments
    /// * `py` - Python interpreter context.
    ///
    /// # Returns
    /// A Python array representing the centroids (weights) of the SOM.
    #[getter]
    fn centroids(&self, py: Python) -> Py<PyArray2<f64>> {
        self.centroids.as_ref().unwrap().to_pyarray(py).into()
    }

    /// Computes and returns the quantization error of the SOM.
    /// 
    /// # Returns
    /// The quantization error, which is the sum of the smallest distances between each input and its closest centroid.
    #[getter]
    fn quantization(&self) -> f64 {
        self.dst
            .axis_iter(Axis(0))
            .map(|row| row.iter().cloned().fold(f64::INFINITY, f64::min))
            .sum::<f64>()
            / self.dst.len() as f64
    }


    /// Trains the Self-Organizing Map on the provided data.
    ///
    /// # Arguments
    /// * `x` - A vector of input data, where each element is a feature vector.
    /// * `sigma_initial` - The initial value of the neighborhood radius.
    /// * `epochs` - The number of training epochs.
    /// * `verbose` - A boolean flag to enable/disable verbose output during training.
    ///
    /// # Returns
    /// The trained SOM instance.
    #[pyo3(signature = (x, sigma_initial=1.0, epochs=1, verbose=true))]
    fn fit(&mut self, x: Vec<Vec<f64>>, sigma_initial: f64, epochs: usize, verbose: bool) -> Self {
        let (m, _) = self.units.dim();
        let x = array2(x);

        // Initialize weights randomly from the input data
        self.centroids = Some(sample_rows(&x, m));

        // Precompute distances between units
        let unit_dst = compute_distance_matrix(&self.units, &self.units);

        // Training loop
        for e in 0..epochs {
            let sigma = sigma_initial * E.powf(-(e as f64) / (epochs as f64));

            // Compute distances between inputs and centroids once per epoch
            self.dst = compute_distance_matrix(&x, &self.centroids.as_ref().unwrap());
            if verbose && e % ((epochs as f64 / 100.0) + 1.0) as usize == 0 {
                println!(
                    "Epoch {e:>8}/{epochs:<20} Quantization: {error}",
                    e = e,
                    epochs = epochs,
                    error = self.quantization()
                )
            }

            self.centroids = Some(organize(&x, &unit_dst, &self.dst, m, sigma))
        }

        self.clone()
    }

    /// Custom string representation of the SOM instance.
    ///
    /// # Returns
    /// A string representation of the SOM, including the number of units and the quantization error.
    fn __repr__(&self) -> String {
        format!(
            "SOM(units={units}, quantization={quant})",
            units=self.units.nrows(), quant=self.quantization()
        )
    }
}

#[pymodule]
mod somu {
    #[pymodule_export]
    use super::som;

    #[pymodule_export]
    use super::distance_matrix;

    #[pymodule_export]
    use super::SOM;
}