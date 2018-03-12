use core::linear;
use std::os::raw::c_void;

#[no_mangle]
pub unsafe extern fn linear_regress_input(coefficients: *mut Vec<f64>, example: *mut c_void, array_length: i32) -> f64 {
    linear::regression::regress_input(coefficients, example, array_length)
}

#[no_mangle]
pub unsafe extern fn linear_train_regression(coefficients: *mut Vec<f64>, examples: *mut c_void, nb_of_examples: i32, array_length: i32) {
    linear::regression::train_regression(coefficients, examples, nb_of_examples, array_length)
}

#[cfg(test)]
mod tests {
    use public::linear::*;
    use public::linear::regression::*;

    #[test]
    fn test_regression_for_3_inputs() {
        unsafe {
            let model: *mut Vec<f64> = linear_create(3);
            let examples = Box::into_raw(Box::new([1.0, 2.0, 6.0, -1.0, 6.0, 5.0, -1.0, 5.0, 3.0])) as *mut c_void;
            linear_train_regression(model, examples, 3, 9);

            let mut input = Box::into_raw(Box::new([2.0, 6.0])) as *mut c_void;
            let mut result = linear_regress_input(model, input, 2);
            assert!(result < 1.1);
            assert!(result > 0.9);

            input = Box::into_raw(Box::new([6.0, 5.0])) as *mut c_void;
            result = linear_regress_input(model, input, 2);
            assert!(result > -1.1);
            assert!(result < -0.9);

            input = Box::into_raw(Box::new([5.0, 3.0])) as *mut c_void;
            result = linear_regress_input(model, input, 2);
            assert!(result > -1.1);
            assert!(result < -0.9); 
        }
    }

    #[test]
    fn test_regression_for_4_inputs() {
        unsafe {
            let model: *mut Vec<f64> = linear_create(4);
            let examples = Box::into_raw(Box::new([1.0, 2.0, 6.0, -1.0, -1.0, 5.0, -1.0, 5.0, 1.0, -1.0, -5.0, 2.0])) as *mut c_void;
            linear_train_regression(model, examples, 3, 12);

            let mut input = Box::into_raw(Box::new([2.0, 6.0, -1.0])) as *mut c_void;
            let mut result = linear_regress_input(model, input, 3);
            assert!(result < 1.1);
            assert!(result > 0.9);

            input = Box::into_raw(Box::new([5.0, -1.0, 5.0])) as *mut c_void;
            result = linear_regress_input(model, input, 2);
            assert!(result > -1.1);
            assert!(result < -0.9);

            input = Box::into_raw(Box::new([-1.0, -5.0, 2.0])) as *mut c_void;
            result = linear_regress_input(model, input, 2);
            assert!(result < 1.1);
            assert!(result > 0.9);
        }
    }

    

}