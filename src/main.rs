#![feature(const_fn_trait_bound)]

// mod rs_leastsquare;
extern crate nalgebra as na;
mod gauss_newton;
use crate::gauss_newton::gauss_newton;

use std::collections::BTreeMap as HashMap;

// use crate::rs_leastsquare::least_squares; //HashMap;

// not possible, even in unstable???
// const fn const_max<T: Ord+ Copy>(a: T, b: T) -> T {
//     if a>b { a } else {b}
// }

const NUM_GOODS: usize = 4;
const NUM_LABORS: usize = 5;
const OVERPRODUCTION_TARGET: f32 = 1.01;
// const NUM_MAX: usize = 5; //const_max::<usize>(NUM_GOODS,NUM_LABORS);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Good {
    Log,  // Units: Kg
    Wood, // Units Kg
    Meat, // Units: Kg
    Food,
}

const GOODS: [Good; NUM_GOODS] = [Good::Log, Good::Wood, Good::Meat, Good::Food];

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Labor {
    Lumberjack,
    Carpenter,
    Fisher,
    Hunter,
    Cook,
}

const LABORS: [Labor; NUM_LABORS] = [
    Labor::Lumberjack,
    Labor::Carpenter,
    Labor::Fisher,
    Labor::Hunter,
    Labor::Cook,
];

impl Labor {
    fn industry(&self) -> Industry {
        match self {
            Labor::Lumberjack => Industry {
                inputs: &[],
                outputs: &[(Good::Log, 10.0)],
            },
            Labor::Carpenter => Industry {
                inputs: &[(Good::Log, 10.0)],
                outputs: &[(Good::Wood, 10.0)], // 1/3rd is 'wasted' (sawdust, etc.)
            },
            Labor::Fisher => Industry {
                inputs: &[(Good::Wood, 0.1)],
                outputs: &[(Good::Meat, 1.0)],
            },
            Labor::Hunter => Industry {
                inputs: &[],
                outputs: &[(Good::Meat, 1.0)],
            },
            Labor::Cook => Industry {
                inputs: &[(Good::Wood, 0.2), (Good::Meat, 1.0)],
                outputs: &[(Good::Food, 1.0)], // Some fish is wasted (gutting)
            },
        }
    }
}

struct Industry {
    inputs: &'static [(Good, f32)],
    outputs: &'static [(Good, f32)],
}

struct Economy {
    // Economy population
    pop: f32,

    // Number of laborers allocated to each industry
    laborers: HashMap<Labor, f32>,
    // The relative productivity of each labor in the last tick
    // 0.0 = At least one of the required input goods was not available
    // 1.0 = All of the required input goods were available, sufficiently to saturate demand
    // This is the minimum of the proportion that each input was supplied
    productivity: HashMap<Labor, (f32, Option<Good>)>,

    // Given current workforce allocation, how much of each good will be produced on the next tick?
    // This is expressed as a proportion of the total required for industry. i.e:
    // >= 1.0 => supply completely saturates industry, oversupply
    // <= 1.0 => supply is insufficient to satisfy industry, undersupply
    available: HashMap<Good, f32>,

    // Labor value and consumption value are in the same units:
    // - Labor value are the average number of labor hours required to produce 1 unit
    // - Consumption values are the number of labor hours that workers would be willing to exchange for 1 unit
    // During each tick, labor values are propagated forwards through the supply chain and consumption values are
    // propagated backwards through the supply change, accounting for scarcity.
    labor_value: HashMap<Good, f32>,
    // consumption_value: HashMap<Good, f32>,

    // The relative value of goods. Goods that are produced optimally are at 1.0 (i.e: labor value matches consumption value).
    // > 1.0 => production of this good should increase
    // < 1.0 => production of this good should reduce
    // value: HashMap<Good, f32>,
    price: HashMap<Good, f32>,

    // Total output of this good that occured in the last tick
    output: HashMap<Good, f32>,

    demand: HashMap<Good, f32>,
}

fn my_print<const M: usize,const N: usize>(
    y: &nalgebra::SMatrix<f32,M,1>,
    x: &nalgebra::SMatrix<f32,M,N>,
    beta: Option<&nalgebra::SMatrix<f32,N,1>>,
) {
    print!("\t\t");
    if let Some(beta) = beta {
        for j in 0..x.ncols() {
            print!("{:.2}\t", beta[(j, 0)]);
        }
    }
    print!("\n");
    for i in 0..x.nrows() {
        print!("{:.2}\t\t", y[(i, 0)]);
        for j in 0..x.ncols() {
            print!("{:.3}\t", x[(i, j)]);
        }
        print!("\n");
    }
}

// fn newton(
//     y: &nalgebra::DMatrix<f32>,
//     x: &nalgebra::DMatrix<f32>,
//     beta_start: &nalgebra::DMatrix<f32>,
// ) -> nalgebra::DMatrix<f32> {
//     let mut beta = beta_start.clone();
//     let rows = beta.nrows();
//     let f_x = x * beta.clone() - y;
//     //.norm();
//     //let norm = f_x.norm();
//     //dbg!((&beta, &f_x));
//     for i in 0..rows {
//         let sum = x
//             .row(i)
//             .iter()
//             .enumerate()
//             .map(|(r, &val)| 2.0 * f_x[(r, 0)] * val)
//             .sum::<f32>();
//         let scale = f_x[(i, 0)].powi(2);
//         dbg!((scale, sum, -scale / sum));
//         beta[(i, 0)] -= scale / sum;
//     }
//     beta
// }

// fn gradient_descend(
//     y: &nalgebra::DMatrix<f32>,
//     x: &nalgebra::DMatrix<f32>,
//     beta_start: &nalgebra::DMatrix<f32>,
// ) -> nalgebra::DMatrix<f32> {
//     let r = y - x * beta_start.clone();
//     let r_t = r.transpose();
//     let gamma1 = (r_t.clone() * r.clone())[(0, 0)];
//     let gamma2 = (r_t.clone() * (x * r.clone()))[(0, 0)];
//     dbg!(gamma1);
//     dbg!(gamma2);
//     let gamma = gamma1 / gamma2;
//     dbg!(&r);
//     dbg!(gamma);
//     beta_start + gamma * r
// }

impl Economy {
    // Calculate to what extent supply will satisfy demand for each good on the upcoming tick. See Economy::available.
    fn derive_available_goods(&mut self) {
        let mut total_demand = HashMap::new();
        let mut total_supply = HashMap::new();

        for labor in LABORS {
            let industry = labor.industry();

            let laborers = self.laborers.get(&labor).unwrap_or(&0.0);

            // Productivity may limit goods that can be produced if inputs are undersupplied
            // If 1.0, all industry inputs are satisfied. If 0.0, no industry inputs are satisfied.
            let (limiting_good, productivity) = industry
                .inputs
                .iter()
                // Productivity can never be lower than 0% or higher than 100%. You can throw capital at a tree as much
                // as you like: labor is required for economic output!
                .map(|(good, _)| {
                    (
                        Some(*good),
                        self.available.get(good).unwrap_or(&0.0).max(0.0).min(1.0),
                    )
                })
                .min_by_key(|(_, available)| (*available * 100000.0) as i64) // PartialOrd hack
                .unwrap_or((None, 1.0));

            for &(good, input) in industry.inputs {
                *total_demand.entry(good).or_insert(0.0) += input * laborers;
            }

            self.productivity
                .insert(labor, (productivity, limiting_good));

            for &(good, output) in industry.outputs {
                //dbg!(&(good, output, laborers, productivity));
                *total_supply.entry(good).or_insert(0.0) += output * laborers * productivity;
            }
        }

        // TODO: determine required food based on consumption value & Maslow hierachy
        total_demand.insert(Good::Food, self.pop * 0.5);

        for good in GOODS {
            let total_supply = total_supply.get(&good).unwrap_or(&0.0);
            let total_demand = total_demand.get(&good).unwrap_or(&0.0);
            // println!("{:?}, total_supply = {}, total_demand = {}", good, total_supply, total_demand);
            self.available
                .insert(good, total_supply / total_demand.max(0.00001));
            self.price
                .insert(good, total_demand / total_supply.max(0.00001));
            self.demand.insert(good, *total_demand);
        }
    }

    // Calculate labor values for each good by propagating its value forward through the supply chain (this is the easy
    // part). The labor value of each good is simply the sum of the labor values of its inputs, in addition to the
    // labor time required to create a unit of the input.
    //
    // Because more than one industry might produce the same good, we keep a running total of labour values vs outputs
    // so that we can normalise this value across the industries afterwards.
    fn derive_labor_values(&mut self) {
        let mut total_labor_values = HashMap::<Good, f32>::new();
        let mut total_produced = HashMap::<Good, f32>::new();

        for labor in LABORS {
            let industry = labor.industry();

            let laborers = self.laborers.get(&labor).unwrap_or(&0.0);

            let total_input_value = industry
                .inputs
                .iter()
                .map(|(good, input)| *self.labor_value.get(good).unwrap_or(&0.0) * input)
                .sum::<f32>();

            let labor_time = 1.0;

            let productivity = self.productivity.get(&labor).unwrap_or(&(0.0, None)).0;

            for &(good, output) in industry.outputs {
                let volume = output * laborers * productivity;

                *total_labor_values.entry(good).or_insert(0.0) +=
                    (total_input_value + labor_time) / volume;
                *total_produced.entry(good).or_insert(0.0) += volume;
            }
        }

        for good in GOODS {
            let total_labor_value = total_labor_values.get(&good).unwrap_or(&0.0);
            let total_produced = total_produced.get(&good).unwrap_or(&0.0);
            self.labor_value
                .insert(good, total_labor_value / total_produced.max(0.00001));

            self.output.insert(good, *total_produced);
        }
    }

    fn redistribute_laborers(&mut self) {
        // minimize sum of ((supply-demand)/demand)²
        // minimize sum of (supply/demand + BIAS)²
        const BIAS: f32 = -OVERPRODUCTION_TARGET; // bias slightly towards overproduction

        // supply = workers * amount * productivity
        // so https://en.wikipedia.org/wiki/Ordinary_least_squares
        // y = [-BIAS; N]
        // X[n][p] = amount_np * productivity_p/ demand_n
        // beta = laborers: [_;P]

        let y =
            na::SMatrix::<f32,NUM_GOODS, 1>::from_fn(|i, _| if i < NUM_GOODS { -BIAS } else { 0.0 });
        let mut x = na::SMatrix::<f32,NUM_GOODS,NUM_LABORS>::from_fn(|_n, _p| 0.0);
        for p in 0..NUM_LABORS {
            let labor = LABORS[p];
            let products = labor.industry().outputs;
            for (good, amount) in products {
                let n = GOODS.iter().enumerate().find(|x| *x.1 == *good).unwrap().0;
                //dbg!((n, p, amount, self.productivity[&labor].0, self.demand[good]));
                x[(n, p)] = *amount / self.productivity[&labor].0.max(0.1) / self.demand[good];
            }
        }
        // solve the under-determinism by making fisher and hunter scale by their efficiency
        // x[(4, 2)] = -1.0;
        // x[(4, 3)] = x[(2, 2)] / x[(2, 3)];
        let beta_start = na::SMatrix::<f32, NUM_LABORS, 1>::from_fn(|i, _| self.laborers[&LABORS[i]]);
        let mut beta = beta_start;
        for _ in 0..1 {
            beta = gauss_newton(&x, &y, &beta);
            my_print(&y, &x, Some(&beta));
        }
        //for _ in 0..5 { beta = gradient_descend(&y, &x, &beta); my_print(&y, &x, Some(&beta)); }
        // let beta = least_squares(&x, &y);
        // my_print(&y, &x, beta.as_ref());

        if true {
            //let Some(beta) = beta {
            for i in 0..NUM_LABORS {
                self.laborers
                    .get_mut(&LABORS[i])
                    .map(|val| *val = beta[(i, 0)]);
            }
        }

        let working_pop = self.pop * 1.0; // For now, assume everybody in the economy can work (1.0)
        let total_laborers = self.laborers.values().sum::<f32>();

        self.laborers.values_mut().for_each(|l| {
            // This prevents any industry becoming completely drained of workforce, thereby inhibiting any production.
            // Keeping a small number of laborers in every industry keeps things 'ticking over' so the economy can
            // quickly adapt to changing conditions
            let min_workforce_alloc = 0.01;

            let factor = if total_laborers > working_pop {
                working_pop / total_laborers
            } else {
                1.0
            };
            *l = (*l * factor).max(min_workforce_alloc);
        });
    }

    fn tick(&mut self) {
        self.derive_available_goods();
        self.derive_labor_values();
        // self.derive_consumption_values();
        // self.derive_values();
        self.redistribute_laborers();
    }
}

fn main() {
    let mut economy = Economy {
        pop: 100.0,
        laborers: HashMap::new(),
        productivity: HashMap::new(),
        available: HashMap::new(),
        labor_value: HashMap::new(),
        // consumption_value: HashMap::new(),
        // value: HashMap::new(),
        price: HashMap::new(),
        output: HashMap::new(),
        demand: HashMap::new(),
    };

    economy.laborers.insert(Labor::Lumberjack, 1.0);
    economy.laborers.insert(Labor::Carpenter, 1.0);
    economy.laborers.insert(Labor::Fisher, 1.0);
    economy.laborers.insert(Labor::Hunter, 1.0);
    economy.laborers.insert(Labor::Cook, 1.0);

    for i in 0..100
    /*100*/
    {
        println!("--- Tick {} ---", i);
        economy.tick();

        println!(
            "Laborers: {:?} ({}% lazy, pop = {})",
            economy.laborers,
            100.0 * (economy.pop - economy.laborers.values().sum::<f32>()) / economy.pop,
            economy.pop
        );
        println!("Available: {:?}", economy.available);
        // println!("Consumption value: {:?}", economy.consumption_value);
        println!("Labor value: {:?}", economy.labor_value);
        // println!("Value: {:?}", economy.value);
        println!("Price: {:?}", economy.price);
        println!("Demand: {:?}", economy.demand);
        println!("Productivity: {:?}", economy.productivity);
        println!("Total output: {:?}", economy.output);
    }
}
