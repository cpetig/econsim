/// Very small replacement for nalgebra
use core::{
    fmt::{Debug, Formatter},
    iter::Sum,
};
use num_traits::float::Float;
use std::{
    fmt::Write,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub},
};

/// Matrix with static size
#[derive(Clone)]
pub struct SMatrix<T: Float, const R: usize, const C: usize> {
    buffer: [[T; C]; R],
}

/// Vector with static size
#[derive(Clone)]
pub struct SVector<T, const R: usize> {
    buffer: [T; R],
}

/// Column of a matrix
pub struct ColumnRef<'a, T: Float, const R: usize, const C: usize> {
    column: usize,
    matrix: &'a SMatrix<T, R, C>,
}

/// Column iterator into matrix
pub struct ColumnRefIter<'a, T: Float, const R: usize, const C: usize> {
    column: usize,
    row: usize,
    matrix: &'a SMatrix<T, R, C>,
}

impl<T: Float + Default + Sum + AddAssign, const R: usize, const C: usize> SMatrix<T, R, C> {
    pub fn zeros() -> Self {
        SMatrix {
            buffer: [[T::default(); C]; R],
        }
    }

    pub fn from_fn<F: Fn(usize, usize) -> T>(f: F) -> Self {
        let mut res = Self::zeros();
        for r in 0..R {
            for c in 0..C {
                res.buffer[r][c] = f(r, c);
            }
        }
        res
    }

    pub fn row(&self, r: usize) -> &[T; C] {
        &self.buffer[r]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T; C]> {
        self.buffer.iter()
    }

    pub fn column(&self, column: usize) -> ColumnRef<'_, T, R, C> {
        ColumnRef {
            column,
            matrix: &self,
        }
    }

    pub fn norm_squared(&self) -> T {
        self.buffer
            .iter()
            .map(|r| r.iter().map(|x| *x * *x).sum())
            .sum()
    }
}

impl<T: Float + Default + Sum + AddAssign, const R: usize> SVector<T, R> {
    pub fn zeros() -> Self {
        SVector {
            buffer: [T::default(); R],
        }
    }

    pub fn from_fn<F: Fn(usize, usize) -> T>(f: F) -> Self {
        let mut res = Self::zeros();
        for r in 0..R {
            res.buffer[r] = f(r, 0);
        }
        res
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    pub fn zip_map<F: Fn(T, T) -> T>(&self, b: &Self, f: F) -> Self {
        let mut res = Self::zeros();
        for r in 0..R {
            res.buffer[r] = f(self.buffer[r], b.buffer[r]);
        }
        res
    }

    pub fn component_mul(&self, b: &Self) -> Self {
        let mut res = Self::zeros();
        for r in 0..R {
            res.buffer[r] = self.buffer[r] * b.buffer[r];
        }
        res
    }

    pub fn norm_squared(&self) -> T {
        self.buffer.iter().map(|x| *x * *x).sum()
    }
}

impl<T: Float + Default + Sum + AddAssign, const R: usize, const C: usize> Mul<SVector<T, C>>
    for SMatrix<T, R, C>
{
    type Output = SVector<T, R>;

    fn mul(self, rhs: SVector<T, C>) -> Self::Output {
        let mut res = Self::Output::zeros();
        for r in 0..R {
            let mut cres = T::default();
            for c in 0..C {
                cres += self.buffer[r][c] * rhs.buffer[c];
            }
            res.buffer[r] = cres;
        }
        res
    }
}

// perhaps there is a more elegant way to derive this?
impl<T: Float + Default + Sum + AddAssign, const R: usize, const C: usize> Mul<&SVector<T, C>>
    for &SMatrix<T, R, C>
{
    type Output = SVector<T, R>;

    fn mul(self, rhs: &SVector<T, C>) -> Self::Output {
        let mut res = Self::Output::zeros();
        for r in 0..R {
            let mut cres = T::default();
            for c in 0..C {
                cres += self.buffer[r][c] * rhs.buffer[c];
            }
            res.buffer[r] = cres;
        }
        res
    }
}

// scalar multiplication
impl<const R: usize> Mul<SVector<f32, R>> for f32 {
    type Output = SVector<f32, R>;

    fn mul(self, rhs: SVector<f32, R>) -> Self::Output {
        let mut res = Self::Output::zeros();
        for r in 0..R {
            res.buffer[r] = self * rhs.buffer[r];
        }
        res
    }
}

impl<T: Float + Default + Sum + AddAssign, const R: usize> Add<SVector<T, R>> for SVector<T, R> {
    type Output = SVector<T, R>;

    fn add(self, rhs: SVector<T, R>) -> Self::Output {
        let mut res = self;
        for r in 0..R {
            res.buffer[r] += rhs.buffer[r];
        }
        res
    }
}

impl<T: Float + Default + Sum + AddAssign, const R: usize> Add<&SVector<T, R>> for &SVector<T, R> {
    type Output = SVector<T, R>;

    fn add(self, rhs: &SVector<T, R>) -> Self::Output {
        let mut res = self.clone();
        for r in 0..R {
            res.buffer[r] += rhs.buffer[r];
        }
        res
    }
}

impl<T: Float + Default + Sum + AddAssign, const R: usize, const C: usize> Sub<&SMatrix<T, R, C>>
    for &SMatrix<T, R, C>
{
    type Output = SMatrix<T, R, C>;

    fn sub(self, rhs: &SMatrix<T, R, C>) -> Self::Output {
        let mut res = self.clone();
        for r in 0..R {
            for c in 0..C {
                res.buffer[r][c] += -rhs.buffer[r][c];
            }
        }
        res
    }
}

impl<T: Float, const R: usize, const C: usize> Index<(usize, usize)> for SMatrix<T, R, C> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.buffer[index.0][index.1]
    }
}

impl<T: Float, const R: usize, const C: usize> IndexMut<(usize, usize)> for SMatrix<T, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.buffer[index.0][index.1]
    }
}

impl<T: Float, const R: usize> Index<(usize, usize)> for SVector<T, R> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.buffer[index.0]
    }
}

impl<T: Float, const R: usize> IndexMut<(usize, usize)> for SVector<T, R> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.buffer[index.0]
    }
}

impl<T: Float, const R: usize, const C: usize> Neg for SMatrix<T, R, C> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut res = self;
        for r in 0..R {
            for c in 0..C {
                res.buffer[r][c] = -res.buffer[r][c];
            }
        }
        res
    }
}

impl<T: Float + Default + Sum + AddAssign + Debug, const R: usize, const C: usize> Debug
    for SMatrix<T, R, C>
{
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        fmt.write_str("[\n")?;
        for r in self.iter() {
            for c in r.iter() {
                c.fmt(fmt)?;
                fmt.write_char(' ')?;
            }
            fmt.write_char('\n')?;
        }
        fmt.write_char(']')?;
        Ok(())
    }
}

impl<T: Float + Default + Sum + AddAssign + Debug, const R: usize> Debug for SVector<T, R> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        fmt.write_char('[')?;
        for r in self.iter() {
            r.fmt(fmt)?;
            fmt.write_char(' ')?;
        }
        fmt.write_char(']')?;
        Ok(())
    }
}

impl<'a, T: Float + Default + Sum + AddAssign, const R: usize, const C: usize>
    ColumnRef<'a, T, R, C>
{
    pub fn iter(&self) -> ColumnRefIter<'a, T, R, C> {
        ColumnRefIter {
            column: self.column,
            row: 0,
            matrix: self.matrix,
        }
    }
}

impl<'a, T: Float + Default + Sum + AddAssign, const R: usize, const C: usize> Iterator
    for ColumnRefIter<'a, T, R, C>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.row += 1;
        if self.row > R {
            return None;
        }
        Some(&self.matrix.buffer[self.row - 1][self.column])
    }
}

#[cfg(test)]
mod tests {
    use crate::site::economy::mini_nalgebra::{SMatrix, SVector};
    use core::{cmp::PartialEq, fmt::Debug, iter::Sum};
    use num_traits::float::Float;
    use std::ops::AddAssign;

    impl<T: Float + Default + Sum + AddAssign + Debug, const R: usize, const C: usize>
        PartialEq<[[T; C]; R]> for SMatrix<T, R, C>
    {
        fn eq(&self, other: &[[T; C]; R]) -> bool {
            for r in 0..R {
                for c in 0..C {
                    if self[(r, c)] != other[r][c] {
                        return false;
                    }
                }
            }
            true
        }
    }
    impl<T: Float + Default + Sum + AddAssign + Debug, const R: usize> PartialEq<[T; R]>
        for SVector<T, R>
    {
        fn eq(&self, other: &[T; R]) -> bool {
            for r in 0..R {
                if self[(r, 0)] != other[r] {
                    return false;
                }
            }
            true
        }
    }

    #[test]
    pub fn test_mini_nalgebra() {
        assert_eq!(SMatrix::<f32, 2, 2>::zeros(), [[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(SVector::<f32, 2>::zeros(), [0.0, 0.0]);
        assert_eq!(
            SMatrix::<f32, 2, 2>::from_fn(|r, c| (r as f32) * 2.0 + (c as f32)),
            [[0.0, 1.0], [2.0, 3.0]]
        );
        let a = SMatrix::<f32, 2, 2>::from_fn(|r, c| (r as f32) * 2.0 + (c as f32));
        assert_eq!(a.row(0), &[0.0_f32, 1.0]);
        assert_eq!(
            a.iter().next().map(|r| r.iter().next()).flatten().copied(),
            Some(0.0)
        );
        assert_eq!(
            a.iter().nth(1).map(|r| r.iter().nth(1)).flatten().copied(),
            Some(3.0)
        );
        assert_eq!(
            a.column(1).iter().copied().collect::<Vec<f32>>(),
            vec![1.0, 3.0]
        );
        assert_eq!(a.norm_squared(), 14.0);
        let b = SVector::<f32, 2>::from_fn(|r, _| (r + 1) as f32);
        assert_eq!(b, [1.0, 2.0]);
        let c = SVector::<f32, 2>::from_fn(|r, _| 2.5 - (r as f32));
        assert_eq!(c, [2.5, 1.5]);
        assert_eq!(b.zip_map(&c, |a, b| a * b), [2.5, 3.0]);
        assert_eq!(b.component_mul(&c), [2.5, 3.0]);
        assert_eq!(c.norm_squared(), 8.5);
        assert_eq!(a.clone() * b.clone(), [2.0, 8.0]);
        assert_eq!(&a * &b, [2.0, 8.0]);
        assert_eq!(3.0 * b.clone(), [3.0, 6.0]);
        assert_eq!(b.clone() + c.clone(), [3.5, 3.5]);
        assert_eq!(&b + &c, [3.5, 3.5]);
        let d = SMatrix::<f32, 2, 2>::from_fn(|r, c| ((2 - r) as f32) * 2.0 + ((2 - c) as f32));
        assert_eq!(d, [[6.0, 5.0], [4.0, 3.0]]);
        assert_eq!(&d - &a, [[6.0, 4.0], [2.0, 0.0]]);
        assert_eq!(d[(1, 1)], 3.0);
        assert_eq!(c[(1, 0)], 1.5);
        assert_eq!(-d.clone(), [[-6.0, -5.0], [-4.0, -3.0]]);
    }
}
