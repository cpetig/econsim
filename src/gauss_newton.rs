use std::ptr::eq;

use na::SMatrix;
use rand::Rng;

// see https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
// and https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

extern crate nalgebra as na;

const beta_k: f32 = 0.001; // zero gives GaussNewton, non zero Levenberg-Marquardt

const sqrt2: f32 = 1.414213562_f32; // 2.0_f32.sqrt();

fn f<const M: usize, const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    bias: &na::SMatrix<f32, M, 1>,
    x: &na::SMatrix<f32, N, 1>,
) -> na::SMatrix<f32, M, 1> {
    (equation * x) - bias
}

// df_r(x)/dx_c  (oh it is not dependent on x)
fn J<const M: usize, const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    x: &na::SMatrix<f32, N, 1>,
) -> na::SMatrix<f32, M, N> {
    sqrt2 * equation
}

fn d<const M: usize, const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    bias: &na::SMatrix<f32, M, 1>,
    x: &na::SMatrix<f32, N, 1>,
) -> f32 {
    f(equation, bias, x).norm_squared()
}

fn print<const M: usize, const N: usize>(x: &nalgebra::SMatrix<f32, M, N>) {
    for i in 0..M {
        for j in 0..N {
            print!("{:.3}\t", x[(i, j)]);
        }
        print!("\n");
    }
}

fn inv_recurse<const M: usize, const N: usize>(
    res: &mut SMatrix<f32, N, M>,
    x: &SMatrix<f32, M, N>,
    row: usize,
    destrow: usize,
    factor: f32,
    depth: u32,
) {
    let r = x.row(row);
    let sum: f32 = r
        .iter()
        .filter_map(|a| if *a > 0.0 { Some(*a) } else { None })
        .sum();
    if sum > 0.0 {
        let factor = factor / sum;
        for (col, _) in r.iter().enumerate().filter(|(_, val)| **val > 0.0) {
            res[(col, destrow)] += factor;
            // println!("{} {} {} {}", depth, col, row, factor);
            if depth >= 1 {
                for (row2, value) in x.column(col).iter().enumerate().filter(|(_, v)| **v < 0.0) {
                    inv_recurse(res, x, row2, destrow, -value * factor, depth - 1);
                }
            }
        }
    }
}

fn my_inverse<const M: usize, const N: usize>(
    x: &nalgebra::SMatrix<f32, M, N>,
) -> nalgebra::SMatrix<f32, N, M> {
    let mut res = SMatrix::zeros();
    for row in 0..M {
        inv_recurse(&mut res, &x, row, row, 1.0, 5);
    }
    res
}

pub fn gauss_newton<const M: usize, const N: usize>(
    equation: &na::SMatrix<f32, M, N>,
    bias: &na::SMatrix<f32, M, 1>,
    x0: &na::SMatrix<f32, N, 1>,
) -> na::SMatrix<f32, N, 1> {
    let J = J(equation, x0);
    let JT = J.transpose();
    let I = SMatrix::<f32, N, N>::from_fn(|r, c| if r == c { 1.0 } else { 0.0 });
    let D = JT.clone() * J.clone() + beta_k * I;
    let Dinv = D.try_inverse().unwrap();
    let f_x0 = f(equation, bias, x0);
    let error0 = f_x0.norm_squared();
    let dvec = -(Dinv * (JT * f_x0));
    let scale2 = -(Dinv * JT);
    print(&x0.transpose());
    print(&f_x0.transpose());
    print(&scale2);
    print(&dvec.transpose());
    print(&(scale2 * f_x0).transpose());
    print(&equation);
    let minv = my_inverse(&equation);
    print(&(minv * -1.0 / sqrt2));
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
    x1
}
