use core::multi_layer_perceptron;
use core::multi_layer_perceptron::{MLPType, MLP};
use std::os::raw::c_void;

// Create a model for a MLP, first argument is an array with the lenght being the number of layers and value number of neurons
#[no_mangle]
pub unsafe extern "C" fn MLP_create(neurons_array: *mut c_void, length: i32) -> *mut MLP { // TRY: return *mut c_void
    multi_layer_perceptron::create_model(neurons_array, length)
}

#[no_mangle]
pub extern fn mlp_train_classification(mlp: *mut MLP, examples: *mut c_void, nb_of_examples: i32, array_length: i32,
                                       nb_iterations: i32) {
    multi_layer_perceptron::train(mlp, examples, nb_of_examples, array_length, nb_iterations, MLPType::Classification)
}

#[no_mangle]
pub extern fn mlp_train_regression(mlp: *mut MLP, examples: *mut c_void, nb_of_examples: i32, array_length: i32,
                                       nb_iterations: i32) {
    multi_layer_perceptron::train(mlp, examples, nb_of_examples, array_length, nb_iterations, MLPType::Regression)
}

#[no_mangle]
pub unsafe extern fn mlp_classify(mlp: *mut MLP, input: *mut c_void, array_length: i32) -> f64 {
    multi_layer_perceptron::mlp_get_value_for_inputs(mlp, input, array_length, MLPType::Classification)
}

#[no_mangle]
pub unsafe extern fn mlp_regress(mlp: *mut MLP, input: *mut c_void, array_length: i32) -> f64 {
    multi_layer_perceptron::mlp_get_value_for_inputs(mlp, input, array_length, MLPType::Regression)
}

#[cfg(test)]
mod tests {
    use public::multi_layer_perceptron::*;

    #[test]
    fn test_perceptron_xor_classification() {
        unsafe {
            let neuron_layers = Box::into_raw(Box::new([2, 3, 1])) as *mut c_void;
            let model = MLP_create(neuron_layers, 3);
            let examples = Box::into_raw(Box::new([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])) as *mut c_void;
            mlp_train_classification(model, examples, 4, 12, 80000);

            let mut input = Box::into_raw(Box::new([1.0, 1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) > 0.95);
            assert!(mlp_classify(model, input, 2) < 1.05);

            input = Box::into_raw(Box::new([-1.0, -1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) > 0.95);
            assert!(mlp_classify(model, input, 2) < 1.05);

            input = Box::into_raw(Box::new([1.0, -1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) < -0.95);
            assert!(mlp_classify(model, input, 2) > -1.05);

            input = Box::into_raw(Box::new([-1.0, 1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) < -0.95);
            assert!(mlp_classify(model, input, 2) > -1.05);
        }  
    }

     #[test]
    fn test_perceptron_circle_classification() {
        unsafe {
            let neuron_layers = Box::into_raw(Box::new([2, 10, 10, 1])) as *mut c_void;
            let model = MLP_create(neuron_layers, 4);
            let examples = Box::into_raw(Box::new([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 
                                              1.0, 0.5, 0.5, 1.0, 0.5, -0.5, 1.0, -0.5, 0.5, 1.0, -0.5, -0.5])) as *mut c_void;
            mlp_train_classification(model, examples, 8, 24, 80000);

            let mut input = Box::into_raw(Box::new([1.0, 1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) < -0.5);
            assert!(mlp_classify(model, input, 2) > -1.50);

            input = Box::into_raw(Box::new([-1.0, -1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) < -0.5);
            assert!(mlp_classify(model, input, 2) > -1.50);
            
            input = Box::into_raw(Box::new([1.0, -1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) < -0.5);
            assert!(mlp_classify(model, input, 2) > -1.50);

            input = Box::into_raw(Box::new([-1.0, 1.0])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) < -0.5);
            assert!(mlp_classify(model, input, 2) > -1.50);

            input = Box::into_raw(Box::new([0.5, 0.5])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) > 0.5);
            assert!(mlp_classify(model, input, 2) < 1.50);

            input = Box::into_raw(Box::new([-0.5, -0.5])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) > 0.5);
            assert!(mlp_classify(model, input, 2) < 1.50);

            input = Box::into_raw(Box::new([0.5, -0.5])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) > 0.5);
            assert!(mlp_classify(model, input, 2) < 1.50);

            input = Box::into_raw(Box::new([-0.5, 0.5])) as *mut c_void;
            assert!(mlp_classify(model, input, 2) > 0.5);
            assert!(mlp_classify(model, input, 2) < 1.50);
        }  
    }

}