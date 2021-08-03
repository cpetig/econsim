extern crate nalgebra as na;

const equation: [[f32; 2]; 2] = [[2.0_f32, 1.0], [3.0, 0.0]];
const bias: [f32; 2] = [3.0, 3.0];
const sqrt2: f32 = 1.414213562_f32; // 2.0_f32.sqrt();

fn f(x: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    let xmat = na::DMatrix::<f32>::from_fn(2, 2, |r, c| equation[r][c]);
    let y = na::DMatrix::from_column_slice(2, 1, &bias);
    y - xmat * x
}

fn d(x: &na::DMatrix<f32>) -> f32 {
    f(x).norm_squared()
}

pub fn main() {
    let J = na::DMatrix::from_row_slice(2, 2, &[2.0 * sqrt2, 3.0 * sqrt2, sqrt2, 0.0]);
    let JT = J.transpose();
    let D = JT.clone() * J.clone();
    let x0 = na::DMatrix::from_column_slice(2,1,&[0.0,0.0]);
    dbg!(f(&x0));
    dbg!(d(&x0));
    dbg!(&J);
    dbg!(&D);
    let dvec = - dbg!(D.try_inverse().unwrap()) * dbg!(JT * f(&x0));
    let alpha = 1.0_f32;
    let x1 = x0 + alpha * dvec;
    dbg!(&x1);
    dbg!(d(&x1));
}
