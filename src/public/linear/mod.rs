pub mod classification;
pub mod regression;

use core::linear;
use std::os::raw::c_void;

// Create a linear model for either regression or classification
#[no_mangle] 
pub extern "C" fn linear_create(inputs: i32) -> *mut Vec<f64> {
    linear::create_model(inputs)
}

// Get the weight number for a given index
#[no_mangle]
pub unsafe extern fn get_weight(coefficients: *mut Vec<f64>, weight_number: i32) -> f64 {
    linear::get_weight(coefficients, weight_number)
}
