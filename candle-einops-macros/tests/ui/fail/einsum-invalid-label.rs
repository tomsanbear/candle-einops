use candle_einops_macros::einsum;

fn main() {
    let _ = einsum!("row column$ -> row", ());
}
