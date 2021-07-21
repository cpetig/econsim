mod rs_leastsquare;

use std::collections::BTreeMap as HashMap;//HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Good {
    Log, // Units: Kg
    Wood, // Units Kg
    Meat, // Units: Kg
    Food,
}

const GOODS: [Good; 4] = [
    Good::Log,
    Good::Wood,
    Good::Meat,
    Good::Food,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Labor {
    Lumberjack,
    Carpenter,
    Fisher,
    Hunter,
    Cook,
}

const LABORS: [Labor; 5] = [
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
}

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
            let (limiting_good, productivity) = industry.inputs
                .iter()
                // Productivity can never be lower than 0% or higher than 100%. You can throw capital at a tree as much
                // as you like: labor is required for economic output!
                .map(|(good, _)| (Some(*good), self.available.get(good).unwrap_or(&0.0).max(0.0).min(1.0)))
                .min_by_key(|(_, available)| (*available * 100000.0) as i64) // PartialOrd hack
                .unwrap_or((None, 1.0));

            for &(good, input) in industry.inputs {
                *total_demand.entry(good).or_insert(0.0) += input * laborers;
            }

            self.productivity.insert(labor, (productivity, limiting_good));

            for &(good, output) in industry.outputs {
                *total_supply.entry(good).or_insert(0.0) += output * laborers * productivity;
            }
        }

        // TODO: determine required food based on consumption value & Maslow hierachy
        total_demand.insert(Good::Food, self.pop * 0.5);

        for good in GOODS {
            let total_supply = total_supply.get(&good).unwrap_or(&0.0);
            let total_demand = total_demand.get(&good).unwrap_or(&0.0);
            // println!("{:?}, total_supply = {}, total_demand = {}", good, total_supply, total_demand);
            self.available.insert(good, total_supply / total_demand.max(0.00001));
            self.price.insert(good, total_demand / total_supply.max(0.00001));
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

            let total_input_value = industry.inputs
                .iter()
                .map(|(good, input)| {
                    *self.labor_value.get(good).unwrap_or(&0.0) * input
                })
                .sum::<f32>();

            let labor_time = 1.0;

            let productivity = self.productivity.get(&labor).unwrap_or(&(0.0, None)).0;

            for &(good, output) in industry.outputs {
                let volume = laborers * productivity;

                *total_labor_values
                    .entry(good)
                    .or_insert(0.0) += (total_input_value + labor_time) / output * volume;
                *total_produced.entry(good).or_insert(0.0) += volume;
            }
        }

        for good in GOODS {
            let total_labor_value = total_labor_values.get(&good).unwrap_or(&0.0);
            let total_produced = total_produced.get(&good).unwrap_or(&0.0);
            self.labor_value.insert(good, total_labor_value / total_produced.max(0.00001));

            self.output.insert(good, *total_produced);
        }
    }

    // Calculate the consumption value for each good. This one is a bit more complicated. For each industry, we take a
    // look at the outputs. We sum up the consumption values of the outputs derived from the last tick to get the total
    // output's consumption value of that industry (based on the consumption value of the last tick). Then, we
    // distribute that consumption value to industry inputs. Obviously, we weight this based on the total outputs of
    // each industry.
    // fn derive_consumption_values(&mut self) {
    //     let mut total_consumption_values = HashMap::<Good, f32>::new();
    //     let mut total_used = HashMap::<Good, f32>::new();

    //     for labor in LABORS {
    //         let industry = labor.industry();

    //         let laborers = self.laborers.get(&labor).unwrap_or(&0.0);

    //         let productivity = self.productivity.get(&labor).unwrap_or(&(0.0, None)).0;

    //         let total_output_value = industry.outputs
    //             .iter()
    //             .map(|(good, output)| {
    //                 *self.consumption_value.get(good).unwrap_or(&0.0) * output
    //             })
    //             .sum::<f32>();

    //         let total_inputs = industry.inputs
    //             .iter()
    //             .map(|(_, input)| input)
    //             .sum::<f32>();
    //         let total_outputs = industry.outputs
    //             .iter()
    //             .map(|(_, output)| output)
    //             .sum::<f32>();

    //         for &(good, input) in industry.inputs {
    //             let volume = laborers * productivity * input;

    //             *total_consumption_values
    //                 .entry(good)
    //                 .or_insert(0.0) += total_output_value * input * volume / total_inputs;// / self.available.get(&good).unwrap_or(&0.0).max(0.01);
    //             *total_used.entry(good).or_insert(0.0) += volume * total_outputs;
    //         }
    //     }

    //     for good in GOODS {
    //         let total_consumption_value = total_consumption_values.get(&good).unwrap_or(&0.0);
    //         let total_used = total_used.get(&good).unwrap_or(&0.0);
    //         self.consumption_value.insert(good, total_consumption_value / total_used.max(0.00001));
    //     }

    //     // Automatically derived (TODO: use a Maslow hierarchy to derive the value of consumables)
    //     // This value is "How many hours of labor a citizen would be willing to trade for 1 unit of this good".
    //     // The 'value' of food increases as population increases, and decreases as food production decreases
    //     let food_per_person = self.output.get(&Good::Food).unwrap_or(&0.0) / self.pop;
    //     let starvation = 1.5; // 1.5 hours of work for 1 unit of food when there is no food
    //     let satisfied = 1.0; // Units of food at which value of food falls to 0 no matter how little work is needed
    //     let value_of_food = (1.0 - food_per_person / satisfied).max(0.0) * starvation;
    //     self.consumption_value.insert(Good::Food, 5.0);//1.75 * value_of_food);
    // }

    // Derive relative values for goods from consumption value / labour value. See Economy::value.
    // fn derive_values(&mut self) {
    //     for good in GOODS {
    //         let consumption_value = self.consumption_value.get(&good).unwrap_or(&0.0);
    //         let labor_value = self.labor_value.get(&good).unwrap_or(&0.0);
    //         let availibility = self.available.get(&good).unwrap_or(&0.0).max(0.00001);
    //         println!("Availability of {:?} = {}", good, availibility);
    //         self.value.insert(good, consumption_value / labor_value.max(0.00001) / availibility);
    //     }
    // }

    fn redistribute_laborers(&mut self) {
        // Redistribute labor according to relative values of industry outputs
        for labor in LABORS {
            let industry = labor.industry();

            // let total_consumption_value = industry.outputs
            //     .iter()
            //     .map(|(good, output)| {
            //         *self.consumption_value.get(good).unwrap_or(&0.001) * output
            //     })
            //     .sum::<f32>();
            let total_output = industry.outputs
                .iter()
                .map(|(good, output)| output)
                .sum::<f32>();

            let labor_time = 1.0;
            // Total industry output (used to normalise later, meaningless except in relation to `total_value`)
            let total_labor_value = industry.inputs
                .iter()
                .map(|(good, input)| {
                    *self.labor_value.get(good).unwrap_or(&0.001) * input
                })
                .sum::<f32>() + labor_time;

            // let total_input_values = industry.inputs
            //     .iter()
            //     .map(|(good, input)| {
            //         *self.value.get(good).unwrap_or(&0.001) * input
            //     })
            //     .sum::<f32>();
            // let total_output_values = industry.outputs
            //     .iter()
            //     .map(|(good, output)| {
            //         *self.value.get(good).unwrap_or(&0.001) * output
            //     })
            //     .sum::<f32>();
            let total_input_price = industry.inputs
                .iter()
                .map(|(good, input)| {
                    *self.price.get(good).unwrap_or(&0.001) * input
                })
                .sum::<f32>();
            let total_output_price = industry.outputs
                .iter()
                .map(|(good, output)| {
                    *self.price.get(good).unwrap_or(&0.001) * output
                })
                .sum::<f32>();

            let total_oversupply = industry.outputs
                .iter()
                .map(|(good, output)| {
                    output * self.available.get(&good).unwrap_or(&0.0).max(0.01)
                })
                .sum::<f32>();
            let oversupply = total_oversupply / total_output;

            // If the productivity of an industry is 0, obviously there's no point allocating laborers to it, no matter
            // how valuable the outputs are!
            let (productivity, limiting_good) = *self.productivity.get(&labor).unwrap_or(&(0.0, None));

            // TODO: Use Maslow hierarchy to determine this
            let labor_price = 10.0 / self.price.get(&Good::Food).unwrap_or(&0.0).max(0.00001);

            // Average normalised value of outputs (remember, all values the same = pareto efficiency)
            let avg_value = (total_output_price - total_input_price) * productivity - labor_price;//1.0 / oversupply;//total_consumption_value / total_labor_value.max(0.001);// * productivity;

            // What proportion of the existing workforce should move industries each tick? (at maximum)
            let rate = 0.1;
            let change = (1.0 + avg_value) * rate + (1.0 - rate);
            // println!("change {:?} = {}, avg_value = {}, productivity = {}, total_output_values = {}, total_input_values + labor_time = {}, oversupply = {}, limiting_good = {:?}", labor, change, avg_value, productivity, total_output_values, total_input_values + labor_time, oversupply, limiting_good);
            *self.laborers.entry(labor).or_insert(0.0) *= change;
        }

        // The allocation of the workforce might now be *higher* than the working population! If this is the case, we normalise
        // the workforce over the population to ensure realism is maintained.

        let working_pop = self.pop * 1.0; // For now, assume everybody in the economy can work (1.0)
        let total_laborers = self.laborers.values().sum::<f32>();

        self.laborers.values_mut().for_each(|l| {
            // This prevents any industry becoming completely drained of workforce, thereby inhibiting any production.
            // Keeping a small number of laborers in every industry keeps things 'ticking over' so the economy can
            // quickly adapt to changing conditions
            let min_workforce_alloc = 0.01;

            let factor = if total_laborers > working_pop { working_pop / total_laborers } else { 1.0 };
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
    };

    economy.laborers.insert(Labor::Lumberjack, 1.0);
    economy.laborers.insert(Labor::Carpenter, 1.0);
    economy.laborers.insert(Labor::Fisher, 1.0);
    economy.laborers.insert(Labor::Hunter, 1.0);
    economy.laborers.insert(Labor::Cook, 1.0);

    for i in 0..100 {
        println!("--- Tick {} ---", i);
        economy.tick();

        println!("Laborers: {:?} ({}% lazy, pop = {})", economy.laborers, 100.0 * (economy.pop - economy.laborers.values().sum::<f32>()) / economy.pop, economy.pop);
        println!("Available: {:?}", economy.available);
        // println!("Consumption value: {:?}", economy.consumption_value);
        println!("Labor value: {:?}", economy.labor_value);
        // println!("Value: {:?}", economy.value);
        println!("Price: {:?}", economy.price);
        println!("Total output: {:?}", economy.output);
    }
}
