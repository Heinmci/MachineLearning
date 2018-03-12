pub fn array_to_inputs(inputs: &[f64], input_size: i32) -> Vec<Vec<f64>> {
    let nb_of_inputs = inputs.len() / input_size as usize;
    let mut new_inputs = vec![];

    for i in 0..nb_of_inputs {
        let mut temp = vec![];
        for j in 0..input_size {
            temp.push(inputs[(i*input_size as usize + j as usize) as usize])
        }
        new_inputs.push(temp);
    }
    new_inputs
}