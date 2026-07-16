use candle_einops_macros::einops;

fn main() {
    let _ = einops!("axis -> copy:2 copy:3 axis", ());
}
