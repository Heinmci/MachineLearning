use std::os::raw::c_void;
use std;
use rand::{ Rng};
use rand;
use core::utils;

pub unsafe fn create_model(neurons_array: *mut c_void, length: i32) -> *mut MLP {
    let neurons_per_layer: &[i32];
    neurons_per_layer = std::slice::from_raw_parts(neurons_array as *mut i32, length as usize);
    Box::into_raw(Box::new(MLP::new(neurons_per_layer)))
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum MLPType {
    Classification,
    Regression,
}
#[derive(Debug)]
pub struct MLP {
    pub weights: Vec<Vec<Vec<f64>>>,
    neuron_output: Vec<Vec<f64>>,
    neurons_per_layer: Vec<i32>, // Avec le biais
    sigma_output: Vec<Vec<f64>>
}

impl MLP {
    fn init_weights(neurons_per_layer: &[i32], max: i32) -> Vec<Vec<Vec<f64>>> {
        let mut weights: Vec<Vec<Vec<f64>>> = vec![];
        for _ in 0..neurons_per_layer.len() {
            let mut temp1 = vec![];
            for _ in 0..max {
                let mut temp2 = vec![];
                for _ in 0..max {
                    temp2.push(rand::thread_rng().gen_range(-1.0, 1.0));
                }
                temp1.push(temp2);
            }
            weights.push(temp1);
        }
        weights

    }

    fn init_2_d_vecs(neurons_per_layer: &[i32]) -> Vec<Vec<f64>> {
        let mut vector_2d: Vec<Vec<f64>> = vec![];

        for nb_neurons in neurons_per_layer {
            let mut temp1: Vec<f64> = vec![];
            for _ in 0..*nb_neurons {
                temp1.push(0.0);
            }
            vector_2d.push(temp1);
        }
        vector_2d
    }

    pub fn new(neurons_per_layer: &[i32]) -> MLP {
        let mut max: i32 = 0;
        let mut neurons_per_layer = neurons_per_layer.to_vec();
        for nb in neurons_per_layer.iter_mut() {
            *nb += 1;
            if *nb > max {
                max = *nb;
            }
        }

        let weights = MLP::init_weights(&neurons_per_layer, max);
        let neuron_output = MLP::init_2_d_vecs(&neurons_per_layer);
        let sigma_output = MLP::init_2_d_vecs(&neurons_per_layer);

        MLP {
            weights,
            neuron_output,
            neurons_per_layer: neurons_per_layer,
            sigma_output
        }

    }

    pub fn train_for_one_example(&mut self, input: &Vec<f64>, mlp_type: MLPType) {
        self.init_x_output_for_biais();
        for i in 1..input.len() {
            self.neuron_output[0][i] = input[i];
        }

        for i in 1..self.neurons_per_layer.len() {
            let nb_neurons_for_layer = self.neurons_per_layer[i];
            for j in 1..nb_neurons_for_layer {
                self.neuron_output[i][j as usize] = self.calculate_neuron_output(i as i32,j, mlp_type);
            }
        }
        self.get_sigma_for_last_layer(input[0], mlp_type);
        self.get_sigma_for_other_layers();
        self.update_weights();
    }

    fn update_weights(&mut self) {
        for i in 1..self.neurons_per_layer.len() {
            let neurons_in_layer = self.neurons_per_layer[i];
            let neurons_in_previous_layer = self.neurons_per_layer[i-1];
            for j in 0..neurons_in_previous_layer {
                for k in 0..neurons_in_layer {
                    let left = self.weights[i][j as usize][k as usize];
                    let right = 0.1 * self.neuron_output[i-1][j as usize] * self.sigma_output[i][k as usize];
                    self.weights[i][j as usize][k as usize] = left - right;
                }
            }
        }
    }

    fn get_sigma_for_other_layers(&mut self) { 
        for i in (1..(self.neurons_per_layer.len() - 1)).rev() {
            for j in 0..self.neurons_per_layer[i] {
                let left = 1. - self.neuron_output[i][j as usize].powf(2.0);
                let mut right = 0.;
                for k in 1..self.neurons_per_layer[i+1] {
                    right+= self.weights[i+1][j as usize][k as usize] * self.sigma_output[i+1][k as usize];
                }
                self.sigma_output[i][j as usize] = left * right;
            }
        }
    }

    fn get_sigma_for_last_layer(&mut self, expected_y: f64, mlp_type: MLPType) {
        let nb_neurons_for_last_layer = self.neurons_per_layer.last().unwrap();
        let layer = self.neurons_per_layer.len() - 1;
        for i in 0..*nb_neurons_for_last_layer {
            let output_for_current_neuron = self.neuron_output[layer][i as usize];
            let right = output_for_current_neuron - expected_y;

            if mlp_type == MLPType::Regression {
                self.sigma_output[layer][i as usize] = right;
            } else {
                let left = 1. - output_for_current_neuron.powf(2.0);
                self.sigma_output[layer][i as usize] = left * right;
            }
        }
    }

    fn init_x_output_for_biais(&mut self) {
        for i in 0..self.neurons_per_layer.len() {
            self.neuron_output[i][0] = 1.0;
        }
    }

    fn calculate_neuron_output(&self, layer_number: i32, index_in_current_layer: i32, mlp_type: MLPType) -> f64 {
        let layer_number = layer_number as usize;
        let nb_neurons_from_previous_layer = self.neurons_per_layer[(layer_number -1) as usize];
        let mut sum = 0.0;
        for i in 0..nb_neurons_from_previous_layer {
            let weight = self.weights[layer_number][i as usize][index_in_current_layer as usize];
            let x_output = self.neuron_output[layer_number -1][i as usize];
            sum += weight * x_output;
        }
        
        if layer_number == self.neurons_per_layer.len() - 1 && mlp_type == MLPType::Regression {
            sum
        } else {
            (1. - (-2. * sum).exp()) / (1. + (-2. * sum).exp())
        }
    }
}

pub fn train(mlp: *mut MLP, inputs_array: *mut c_void, nb_of_examples: i32, array_length: i32, nb_iterations: i32, mlp_type: MLPType) {
    let mut remaining_iterations = nb_iterations;
    let mut current_index = 0;
    let input_size = array_length / nb_of_examples;
    let new_arr: &[f64];
    unsafe {
        new_arr = std::slice::from_raw_parts(inputs_array as *mut f64, array_length as usize);
    }
    
    let inputs = utils::array_to_inputs(new_arr, input_size);
    loop {
        if current_index >= nb_of_examples {
            current_index = 0;
        }
    
        if remaining_iterations == 0 {
            break;
        }

        let input = &inputs[current_index as usize];
        unsafe {
            (*mlp).train_for_one_example(input, mlp_type);
        }
        
        remaining_iterations -= 1;
        current_index += 1;
    }
}

pub unsafe fn mlp_get_value_for_inputs(mlp: *mut MLP, input: *mut c_void, array_length: i32, mlp_type: MLPType) -> f64 {
    let input_slice: &[f64];
    input_slice = std::slice::from_raw_parts(input as *mut f64, array_length as usize);
    (*mlp).init_x_output_for_biais();

    for i in 0..input_slice.len() {
        (*mlp).neuron_output[0][i+1] = input_slice[i];
    }

    for i in 1..(*mlp).neurons_per_layer.len() {
        let nb_neurons_for_layer = (*mlp).neurons_per_layer[i];
        for j in 1..nb_neurons_for_layer {
            (*mlp).neuron_output[i][j as usize] = (*mlp).calculate_neuron_output(i as i32,j, mlp_type);
        }
    }
    (*mlp).neuron_output[(*mlp).neurons_per_layer.len() - 1][1]
}

#[cfg(test)]
mod tests {
   /* 
    #[test]
    fn test_perceptron_4_inputs() {
        unsafe {
            let neuron_layers = Box::into_raw(Box::new([4, 3, 1])) as *mut c_void;
            let bob = create_MLP(neuron_layers, 3);
            let bab = Box::into_raw(Box::new([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])) as *mut c_void;
            MLP::mlp_train_classification(bob, bab, 3, 12, 80000);
            
        }  
    }
    #[test]
    fn test_perceptron_8_inputs() {
        unsafe {
            let neuron_layers = Box::into_raw(Box::new([8, 10, 10, 1])) as *mut c_void;
            let bob = create_MLP(neuron_layers, 4);
            let bab = Box::into_raw(Box::new([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])) as *mut c_void;
            MLP::mlp_train_classification(bob, bab, 2, 16, 80000);
            
        }  
    }*/
}