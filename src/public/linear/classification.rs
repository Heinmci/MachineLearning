use core::linear;
use std::os::raw::c_void;

// Return the value (-1 or 1) for given input parameters
#[no_mangle]
pub extern fn linear_classify_input(coefficients: *mut Vec<f64>, input: *mut c_void, array_length: i32) -> f64 {
    linear::classification::classify_input(coefficients, input, array_length)
}

// Train with the given examples, returns 1 if all exampels were calssified 0 if we reached end of iterations
#[no_mangle]
pub extern fn linear_train_classification(coefficients: *mut Vec<f64>, examples: *mut c_void, nb_of_examples: i32, 
                                          array_length: i32) -> i32 {
    linear::classification::train_classification(coefficients, examples, nb_of_examples, array_length)
}

#[cfg(test)]
mod tests {
    use public::linear::*;
    use public::linear::classification::*;

    #[test]
    fn test_classification_for_3_inputs_3_examples() {
        let model: *mut Vec<f64> = linear_create(3);
        let examples = Box::into_raw(Box::new([1.0, 2.0, 6.0, -1.0, 6.0, 5.0, -1.0, 5.0, 3.0])) as *mut c_void;
        let result = linear_train_classification(model, examples, 3, 9);
        assert_eq!(result, 1);

        let mut input = Box::into_raw(Box::new([2.0, 6.0])) as *mut c_void;
        assert_eq!(linear_classify_input(model, input, 2), 1.);

        input = Box::into_raw(Box::new([6.0, 5.0])) as *mut c_void;
        assert_eq!(linear_classify_input(model, input, 2), -1.);

        input = Box::into_raw(Box::new([5.0, 3.0])) as *mut c_void;
        assert_eq!(linear_classify_input(model, input, 2), -1.);  
    }

    #[test]
    fn test_classification_for_3_inputs_4_examples() {
        let modele: *mut Vec<f64> = linear_create(3);
        let examples = Box::into_raw(Box::new([1.0, 2.0, 6.0, -1.0, 6.0, 5.0, -1.0, 5.0, 3.0, -1.0, 5.0, 2.0])) as *mut c_void;
        let result = linear_train_classification(modele, examples, 4, 12);
        assert_eq!(result, 1);

        let mut input = Box::into_raw(Box::new([2.0, 6.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), 1.);

        input = Box::into_raw(Box::new([6.0, 5.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), -1.);

        input = Box::into_raw(Box::new([5.0, 3.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), -1.); 

        input = Box::into_raw(Box::new([5.0, 2.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), -1.);  
    }

    #[test]
    fn test_classification_for_4_inputs_3_examples() {
        let modele: *mut Vec<f64> = linear_create(4);
        let examples = Box::into_raw(Box::new([1.0, 2.0, 2.0, 1.0, -1.0, -5.0, -5.0, -2.0, 1.0, -1.0, 5.0, 2.0])) as *mut c_void;
        let result = linear_train_classification(modele, examples, 3, 12);
        assert_eq!(result, 1);

        let mut input = Box::into_raw(Box::new([2.0, 2.0, 1.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), 1.);

        input = Box::into_raw(Box::new([-5.0, -5.0, -2.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), -1.);

        input = Box::into_raw(Box::new([-1.0, 5.0, 2.0])) as *mut c_void;
        assert_eq!(linear_classify_input(modele, input, 2), 1.);  
    }
}
 