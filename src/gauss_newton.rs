use std::ptr::eq;

use na::SMatrix;
use rand::Rng;

// see https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
// and https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

extern crate nalgebra as na;

// const M: usize = 2; // number of equations
// const N: usize = 3; // number of parameters
const beta_k: f32 = 0.001; // zero gives GaussNewton, non zero Levenberg-Marquardt

const sqrt2: f32 = 1.414213562_f32; // 2.0_f32.sqrt();

fn f<const M: usize,const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    bias: &na::SMatrix<f32, M, 1>,
    x: &na::SMatrix<f32, N, 1>,
) -> na::SMatrix<f32, M, 1> {
    (equation * x) - bias
}

// df_r(x)/dx_c  (oh it is not dependent on x)
fn J<const M: usize,const N: usize>(equation: &na::SMatrix<f32, M, N>, x: &na::SMatrix<f32, N, 1>) -> na::SMatrix<f32, M, N> {
    sqrt2 * equation
//   na::SMatrix::<f32, M, N>::from_fn(|r, c| (sqrt2 * equation[(r, c)]))
}

fn d<const M: usize,const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    bias: &na::SMatrix<f32, M, 1>,
    x: &na::SMatrix<f32, N, 1>,
) -> f32 {
    f(equation, bias, x).norm_squared()
}

fn print<const M: usize,const N: usize>(
    x: &nalgebra::SMatrix<f32,M,N>,
) {
    for i in 0..M {
        for j in 0..N {
            print!("{:.3}\t", x[(i, j)]);
        }
        print!("\n");
    }
}

pub fn gauss_newton<const M: usize,const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    bias: &na::SMatrix<f32, M, 1>,
    x0: &na::SMatrix<f32, N, 1>,
) -> na::SMatrix<f32, N, 1> {
    let J = J(equation, x0);
    let JT = J.transpose();
    let I = SMatrix::<f32, N, N>::from_fn(|r, c| if r == c { 1.0 } else { 0.0 });
    let D = JT.clone() * J.clone() + beta_k * I;
    let Dinv = D.try_inverse().unwrap();
    // print_2x2(&JT);
    // print_2x2(&J);
    // print_2x2(&D);
    //    dbg!(&D);
    // dbg!(f(&x0));
    // dbg!(d(&x0));
    // dbg!(&J);
    // dbg!(&D);
    let f_x0 = f(equation, bias, x0);
    let error0 = f_x0.norm_squared();
    let dvec = -(Dinv * (JT * f_x0));
    let scale2 = -(Dinv * JT);
    print(&x0.transpose());
    print(&f_x0.transpose());
    print(&scale2);
    print(&dvec.transpose());
    print(&(scale2*f_x0).transpose());
    let mut alpha = 1.0_f32;
    // line search
    let x1 = loop {
        let x1 = x0 + alpha * dvec.clone();
        let f_x1 = f(equation, bias, &x1);
        let error1 = f_x1.norm_squared();
        if error1 < error0 {
            break x1;
        }
        alpha /= 2.0;
        if alpha < 0.001 {
            break x0.clone(); // give up
        }
    };
    // dbg!(&x1);
    // dbg!(d(&x1));
    x1
}
