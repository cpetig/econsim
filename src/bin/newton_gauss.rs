// see https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

extern crate nalgebra as na;

const M: usize = 2;
const N: usize = 2;

const equation: [[f32; N]; M] = [[2.0_f32, 1.0], [3.0, 0.0]];
const bias: [f32; M] = [3.0, 3.0];
const sqrt2: f32 = 1.414213562_f32; // 2.0_f32.sqrt();

fn f(x: &na::SMatrix<f32, N, 1>) -> na::SMatrix<f32, M, 1> {
    let xmat = na::SMatrix::<f32, M, N>::from_fn(|r, c| equation[r][c]);
    let y = na::SMatrix::<f32, M, 1>::from_column_slice(&bias);
    (xmat * x) - y
}

// df_r(x)/dx_c  (oh it is not dependent on x)
fn J(x: &na::SMatrix<f32, N, 1>) -> na::SMatrix<f32, M, N> {
    na::SMatrix::<f32, M, N>::from_fn(|r, c| (sqrt2 * equation[r][c]))
}

fn d(x: &na::SMatrix<f32, N, 1>) -> f32 {
    f(x).norm_squared()
}

fn gauss_newton(x0: &na::SMatrix<f32, N, 1>) -> na::SMatrix<f32, N, 1> {
    let J = J(&x0);
    let JT = J.transpose();
    let D = JT.clone() * J.clone();
    let Dinv = D.try_inverse().unwrap();
    // print_2x2(&JT);
    // print_2x2(&J);
    // print_2x2(&D);
    //    dbg!(&D);
    // dbg!(f(&x0));
    // dbg!(d(&x0));
    // dbg!(&J);
    // dbg!(&D);
    let f_x0 = f(&x0);
    let error0 = f_x0.norm_squared();
    let dvec = -(Dinv * (JT * f_x0));
    let mut alpha = 1.0_f32;
    // line search
    let x1 = loop {
        let x1 = x0 + alpha * dvec.clone();
        let f_x1 = f(&x1);
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

pub fn main() {
    let mut x0 = na::SMatrix::<f32, N, 1>::from_column_slice(&[10.4, 0.4]);
    for _ in 0..5 {
        x0 = gauss_newton(&x0);
        println!("[{} {}] {}", x0[(0, 0)], x0[(1, 0)], d(&x0));
    }
}
