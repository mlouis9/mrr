//! Run the PUa-1-0-SL benchmark and write per-cycle snapshots as CSV.
//!
//! ```text
//! cargo run --release -p mrr-slab-1d --example sood -- [OUT_DIR] [N_ATOMS] [CYCLES] [SEED]
//! ```
//!
//! Defaults: `out/sood`, 500 atoms, 50 cycles, seed 1. Writes
//!
//! - `atoms.csv`: `cycle,z,w,q0` — one row per atom per cycle; cycle 0 is the
//!   initial source. `q0` is the group-0 ℓ=0 moment (one-group problem).
//! - `k.csv`: `cycle,k` for cycles ≥ 1.
//! - `meta.csv`: `key,value` with the slab bounds and run parameters.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use mrr_core::convergence::MaxCycles;
use mrr_core::{RunConfig, power_iterate_with};
use mrr_slab_1d::problems::benchmarks::{PUA_1_0_SL_K, pua_1_0_sl};

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    match args.get(i) {
        Some(s) => s
            .parse()
            .unwrap_or_else(|_| panic!("could not parse argument {i}: '{s}'")),
        None => default,
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out_dir: PathBuf = arg(&args, 1, PathBuf::from("out/sood"));
    let n_atoms: usize = arg(&args, 2, 500);
    let cycles: usize = arg(&args, 3, 50);
    let seed: u64 = arg(&args, 4, 1);
    fs::create_dir_all(&out_dir)?;

    let spec = pua_1_0_sl();
    let problem = spec.problem();
    let layout = spec.library().layout();
    let cfg = RunConfig {
        rules: vec![Box::new(MaxCycles::new(cycles))],
        n_source_atoms: n_atoms,
        seed,
    };

    let mut atoms_csv = BufWriter::new(File::create(out_dir.join("atoms.csv"))?);
    let mut k_csv = BufWriter::new(File::create(out_dir.join("k.csv"))?);
    writeln!(atoms_csv, "cycle,z,w,q0")?;
    writeln!(k_csv, "cycle,k")?;

    // The callback cannot return an error, so remember the first one
    let mut io_error: Option<io::Error> = None;
    let result = power_iterate_with(&problem, &cfg, |snap| {
        if io_error.is_some() {
            return;
        }
        let mut write = || -> io::Result<()> {
            let atoms = snap.atoms;
            for i in 0..atoms.len() {
                let q0 = atoms.atom_moments(i)[layout.index(0, 0)];
                writeln!(
                    atoms_csv,
                    "{},{:.17e},{:.17e},{:.17e}",
                    snap.cycle,
                    atoms.points()[i],
                    atoms.weights()[i],
                    q0
                )?;
            }
            if let Some(k) = snap.k {
                writeln!(k_csv, "{},{:.17e}", snap.cycle, k)?;
            }
            Ok(())
        };
        if let Err(e) = write() {
            io_error = Some(e);
        }
    });
    if let Some(e) = io_error {
        return Err(e);
    }
    atoms_csv.flush()?;
    k_csv.flush()?;

    let slab = spec.slab();
    let mut meta = BufWriter::new(File::create(out_dir.join("meta.csv"))?);
    writeln!(meta, "key,value")?;
    writeln!(meta, "problem,PUa-1-0-SL")?;
    writeln!(meta, "lower,{}", slab.lower())?;
    writeln!(meta, "upper,{}", slab.upper())?;
    writeln!(meta, "n_atoms,{n_atoms}")?;
    writeln!(meta, "cycles,{}", result.cycles)?;
    writeln!(meta, "seed,{seed}")?;
    writeln!(meta, "k_ref,{PUA_1_0_SL_K}")?;
    meta.flush()?;

    println!(
        "PUa-1-0-SL: {} cycles, final k = {:.5} (reference {PUA_1_0_SL_K}); wrote {}",
        result.cycles,
        result.k,
        out_dir.display()
    );
    Ok(())
}
