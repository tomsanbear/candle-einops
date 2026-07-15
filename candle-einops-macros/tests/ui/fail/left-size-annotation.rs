use candle_einops_macros::einops;

fn main() {
    let _ = einops!("a:3 b -> b a", ());
}
