#![allow(clippy::unused_unit)]
use polars::prelude::*;
use serde::Deserialize;
use std::cmp;

use pyo3_polars::derive::polars_expr;

fn mix(hash: u64) -> u64 {
    let mut new_hash = hash;
    new_hash ^= new_hash >> 23;
    new_hash = new_hash.wrapping_mul(0x2127599bf4325c37);
    new_hash ^= new_hash >> 47;
    new_hash
}

fn make_masks() -> [u64; 8] {
    let mut masks = [0; 8];
    let mut m = 0u64;
    for mask in &mut masks[..] {
        m = m << 8;
        m |= 255;
        *mask = m;
    }
    masks
}

fn compute_minhash(text: &str, seed: u64) -> u32 {
    let mut min = u64::MAX;
    let m: u64 = 0x880355f21e6d1965;
    let mut buf = 0u64;
    let masks = make_masks();
    for c in text.bytes() {
        buf = buf << 8;
        buf = buf | (c as u64);
        for mask in &masks[..] {
            let ngram = buf & mask;
            let mut hash = seed ^ m;
            hash ^= mix(ngram);
            hash = hash.wrapping_mul(m);
            min = cmp::min(hash, min);
        }
    }
    min.wrapping_sub(min >> 32) as u32
}

#[derive(Deserialize)]
struct MinhashKwargs {
    seed: u64,
}


#[polars_expr(output_type=UInt32)]
fn minhash(inputs: &[Series], kwargs: MinhashKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: UInt32Chunked = ca.apply_values_generic(|value| {
        compute_minhash(value, kwargs.seed)
    });
    Ok(out.into_series())
}
