use itertools::Itertools;
use ndarray::Array2;

pub fn array2<T>(vec: Vec<Vec<T>>) -> Array2<T>
where
    T: Clone,
{
    let flattened: Vec<T> = flatten_vec(&vec);
    Array2::from_shape_vec((vec.len(), vec[0].len()), flattened).expect("Failed to create ArrayD")
}

fn flatten_vec<T>(vec: &Vec<Vec<T>>) -> Vec<T>
where
    T: Clone,
{
    vec.iter().flat_map(|x| x.iter().cloned()).collect()
}

pub fn generate_grid(dims: Vec<usize>) -> Vec<Vec<f64>> {
    dims.into_iter()
        .map(|dim_size| (0..dim_size).map(|x| x as f64))
        .multi_cartesian_product()
        .collect()
}
