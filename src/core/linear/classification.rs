use std::os::raw::c_void;
use std;
use core::utils;

pub fn classify_input(coefficients: *mut Vec<f64>, input: *mut c_void, array_length: i32) -> f64 {
    let input_slice: &[f64];
    unsafe {
        input_slice = std::slice::from_raw_parts(input as *mut f64, array_length as usize);
    }

    calculate_sign(coefficients, input_slice)
}

pub fn train_classification(coefficients: *mut Vec<f64>, inputs: *mut c_void, nb_of_examples: i32, array_length: i32) -> i32 {
    let mut remaining_iterations = 10_000_000;
    let step = 0.1;
    let mut current_index = 0;
    let mut correct_in_a_row = 0;
    let input_size = array_length / nb_of_examples;
    let inputs_slice: &[f64];
    unsafe {
        inputs_slice = std::slice::from_raw_parts(inputs as *mut f64, array_length as usize);
    }
    let inputs = utils::array_to_inputs(inputs_slice, input_size);
    loop {
        if current_index >= nb_of_examples {
            current_index = 0;
        }
        if remaining_iterations == 0 || correct_in_a_row == nb_of_examples{
            break;
        }
        let input = &inputs[current_index as usize];
        let expected_y = input[0];
        let input_sign = calculate_sign(coefficients, &input[1..]); // We don't give the first value because it's our expected output
        if input_sign == expected_y {
            current_index += 1;
            remaining_iterations -= 1;
            correct_in_a_row += 1;
            continue;
        }

        correct_in_a_row = 0;
        
        unsafe {
            (*coefficients)[0] += step * (expected_y - input_sign) as f64;
            for i in 1..input_size {
                (*coefficients)[i as usize] += step * (expected_y - input_sign) as f64 * input[i as usize];
            }
        }
        
        remaining_iterations -= 1;
        current_index += 1;
    }

    if remaining_iterations == 0 {
        0
    } else {
        1
    }
}


fn calculate_sign(coefficients: *mut Vec<f64>, input: &[f64]) -> f64 {
    let mut sign: f64 = 0.;
     unsafe {
         for i in 0..input.len() {
             sign += (*coefficients)[i+1] * input[i];
         }
        sign += (*coefficients)[0];
    };
    if sign < 0.0 {
        -1.
    } else {
        1.
    }
}