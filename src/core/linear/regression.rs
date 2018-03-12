use nalgebra::{DMatrix};
use std::os::raw::c_void;
use std;
use core::linear;
use core::utils;


pub unsafe fn regress_input(coefficients: *mut Vec<f64>, input: *mut c_void, array_length: i32) -> f64 {
    if array_length == 0 {
        return 0.;
    }
    let input_slice: &[f64];
    input_slice = std::slice::from_raw_parts(input as *mut f64, array_length as usize);
    let mut result: f64 = (*coefficients)[0];
    for i in 0..input_slice.len() {
        result += (*coefficients)[i as usize + 1] * input_slice[i];
    }
    result
}

pub unsafe fn train_regression(coefficients: *mut Vec<f64>, inputs: *mut c_void, nb_of_examples: i32, array_length: i32) {
    let input_size = array_length / nb_of_examples;
    let inputs_slice: &[f64];
    
    inputs_slice = std::slice::from_raw_parts(inputs as *mut f64, array_length as usize);

    let inputs = utils::array_to_inputs(inputs_slice, input_size);
    let matrix_slice = slice_to_matrix_vec(&inputs);
    let known_values_slice = get_known_values_from_slice(inputs);
    
    let matrix_x = DMatrix::from_row_slice(
        input_size as usize, // cols
        nb_of_examples as usize,
        matrix_slice.as_slice() // Col par col
    );

    let matrix_y = DMatrix::from_row_slice(
        1,
        nb_of_examples as usize,
        known_values_slice.as_slice()
    );

    let temp_matrix =   matrix_x.clone() * matrix_x.transpose();
    let temp_matrix = temp_matrix.pseudo_inverse(1e-9); // Merci Aurelien
    let result = matrix_x.transpose() * temp_matrix;
    let result = matrix_y * result;
    
    let calulated_weights = result.data.data();
    
    for i in 0..calulated_weights.len() {
        (*coefficients)[i] = calulated_weights[i];
    }
}

fn slice_to_matrix_vec(inputs: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut points_vec = vec![];
    let columns = inputs[0].len(); // On compte la colonne avec resultat attendu pour l'instant
    let lines = inputs.len(); 

    for _ in 0..lines {
        points_vec.push(1.0);
    }

    for i in 1..columns { // On ignore la premiere valeur de chaque ligne car c'est le resultat attendu
        for j in 0..lines {
            points_vec.push(inputs[j][i]);
        }
    }

    points_vec
}

fn get_known_values_from_slice(inputs: Vec<Vec<f64>>) -> Vec<f64> {
    let mut points_vec = vec![];
    let lines = inputs.len(); 

    for i in 0..lines {
        points_vec.push(inputs[i][0]);
    }

    points_vec
}
