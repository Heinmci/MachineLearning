pub mod classification;
pub mod regression;

use rand::{ Rng};
use rand;

pub fn create_model(inputs: i32) -> *mut Vec<f64> {
    let mut weights = vec![];
    for _ in 0..inputs {
        weights.push(rand::thread_rng().gen_range(-1.0, 1.0))
    }
    Box::into_raw(Box::new(weights))
}

pub unsafe fn get_weight(coefficients: *mut Vec<f64>, weight_number: i32) -> f64 {
    (*coefficients)[weight_number as usize]
}

