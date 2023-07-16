use clap::Parser;
use ode_solvers::dopri5::*;
use ode_solvers::*;
use csv::Writer;
use std::error::Error;
use std::fs::File;
use textplots::{Chart, Plot, Shape};

/// CLI Arguments
#[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
struct Args {
    /// time start
    #[arg(long, default_value_t = 0.0)]
    t_start: f64,
    /// time end
    #[arg(long, default_value_t = 100.0)]
    t_end: f64,
    /// time step
    #[arg(long, default_value_t = 10.0)]
    t_step: f64,
    /// y0, initial pop
    #[arg(long, default_value_t = 100.0)]
    y0: f64,
    /// r, growth rate
    #[arg(long, default_value_t = 0.1)]
    r: f64,
    /// k, carrying capacity
    #[arg(long, default_value_t = 1000.0)]
    k: f64,
    /// rtol
    #[arg(long, default_value_t = 1.0e-10)]
    rtol: f64,
    /// atol
    #[arg(long, default_value_t = 1.0e-10)]
    atol: f64,
    /// output file
    #[arg(long, default_value_t = String::from("output.csv"))]
    output: String,
}

type State = Vector1<f64>;
type Time = f64;

struct Model {
    r: f64,
    k: f64,
}

impl ode_solvers::System<State> for Model {
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        dy[0] = self.r * y[0] * (1.0 - y[0] / self.k);
    }
}

fn save_to_csv<T>(x_out: &Vec<T>, y_out: &Vec<T>, path: String) -> Result<(), Box<dyn Error>>
where
    T: std::fmt::Display,
{
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    // Write column names
    writer.write_record(&["x_out", "y_out"])?;

    // Write data rows
    for (x, y) in x_out.iter().zip(y_out.iter()) {
        writer.write_record(&[x.to_string(), y.to_string()])?;
    }

    writer.flush()?;
    Ok(())
}

fn main() {
    let args = Args::parse();
    let system = Model {r: args.r, k: args.k};
    let t_start = args.t_start;
    let t_end = args.t_end;
    let y0 = State::new(args.y0); // Initial population size

    let mut stepper = Dopri5::new(
        system,     // f: F; Structure implementing the System trait
        t_start,    // x: f64; Initial value of the independent variable (usually time)
        t_end,      // x_end: f64; Final value of the independent variable
        args.t_step,       // dx: f64; Increment in the dense output. This argument has no effect if the output type is Sparse
        y0,         // y: OVector<T, D>; Initial value of the dependent variable(s)
        args.rtol,    // rtol: f64; Relative tolerance used in the computation of the adaptive step size
        args.atol,    // atol: f64: Absolute tolerance used in the computation of the adaptive step size
    );
    let res = stepper.integrate();

    // Handle result
    match res {
        Ok(stats) => {
            // stats.print();
            println!("{:?}", stats);

            // Do something with the output...
            // let path = Path::new("./outputs/kepler_orbit_dopri5.dat"); 
            // save(stepper.x_out(), stepper.y_out(), path);  
            // println!("Results saved in: {:?}", path);
            let y_squeezed: Vec<f64> = stepper.y_out().iter().flatten().cloned().collect();

            println!("{:?}", stepper.x_out());
            println!("{:?}", y_squeezed);

            // Plot to terminal
            let zipped: Vec<(f32, f32)> = stepper.x_out()
                .iter()
                .zip(y_squeezed.iter())
                .map(|(&a, &b)| (a as f32, b as f32))
                .collect();
            Chart::new(180, 60, args.t_start as f32, args.t_end as f32)
                .lineplot(&Shape::Lines(&zipped[..])).display();
            
            // Save to CSV
            if let Err(err) = save_to_csv(&stepper.x_out(), &y_squeezed, args.output) {
                eprintln!("Error: {}", err);
            } else {
                println!("Data saved to disk.");
            }
        },
        Err(_) => println!("An error occured."),
    }
}
