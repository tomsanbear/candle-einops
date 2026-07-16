use candle_einops_macros::einops;

fn main() {
    let _ = einops!("axis -> {copies} {copies} axis", ());
}
