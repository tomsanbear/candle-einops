use candle_einops_macros::einops;

fn main() {
    let _ = einops!(
        "(axis huge:18446744073709551615 two:2) -> axis huge two",
        ()
    );
}
