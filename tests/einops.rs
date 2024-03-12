use candle_core::{Device, IndexOp, Result, Tensor};
use candle_einops::{einops, Backend};

#[test]
fn candle_layers() -> Result<()> {
    let input = Tensor::randn(0.0f32, 1.0, (10, 3, 32, 32), &Device::Cpu)?;

    let output1 = einops!("b c (h max(2)) (w max(2)) -> b c h w", &input);
    let output2 = input.max_pool2d(2)?;

    assert_eq!(
        output1.flatten_all()?.to_vec1::<f32>()?,
        output2.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}

#[test]
fn consistency_checks() -> Result<()> {
    let input = Tensor::arange(0f32, (1 * 2 * 3 * 5 * 7 * 11) as f32, &Device::Cpu)?
        .reshape(&[1, 2, 3, 5, 7, 11]);

    let output = einops!("a b c d e f -> a (b) (c d e) f", &input);
    assert_eq!(
        input
            .flatten(0, input.dims().len() - 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        output
            .flatten(0, output.dims().len() - 1)?
            .flatten_all()?
            .to_vec1::<f32>()?,
    );

    let output1 = einops!("a b c d e f -> f e d c b a", &input);
    let output2 = einops!("f e d c b a -> a b c d e f", &input);
    assert_eq!(output1.dims(), output2.dims());
    assert_eq!(
        output1.flatten_all()?.to_vec1::<f32>()?,
        output2.flatten_all()?.to_vec1::<f32>()?
    );

    let intermediate = einops!("a b c d e f -> (f d) c (e b) a", &input);
    let output = einops!("(f d:5) c (e b:2) a -> a b c d e f", &intermediate);
    assert_eq!(output.dims(), input.dims());
    assert_eq!(
        output.flatten_all()?.to_vec1::<f32>()?,
        input.flatten_all()?.to_vec1::<f32>()?
    );

    let input = Tensor::arange(0f32, (2 * 3 * 4) as f32, &Device::Cpu)?.reshape(&[2, 3, 4]);
    let output = einops!("a b c -> b c a", &input);
    assert_eq!(
        input.i((1, 2, 3))?.flatten_all()?.to_vec1::<f32>()?,
        output.i((2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        input.i((0, 1, 2))?.flatten_all()?.to_vec1::<f32>()?,
        output.i((1, 2, 0))?.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}

macro_rules! test {
    ($pattern1:literal, $pattern2:literal, $tensor:ident) => {
        let output1 = einops!($pattern1, &$tensor);
        let output2 = einops!($pattern2, &$tensor);
        assert_eq!(
            output1.flatten_all()?.to_vec1::<f32>()?,
            output2.flatten_all()?.to_vec1::<f32>()?, "({}) & ({}) failed", $pattern1, $pattern2
        );
    };
    ($(($pattern1:literal, $pattern2:literal)),*, $tensor:ident) => {
        $(test!($pattern1, $pattern2, $tensor);)*
    };
}

#[test]
fn equivalent_rearrange() -> Result<()> {
    let input =
        Tensor::arange(0f32, (2 * 3 * 4 * 5 * 6) as f32, &Device::Cpu)?.reshape(&[2, 3, 4, 5, 6]);
    test![
        ("a b c d e -> (a b) c d e", "a b .. -> (a b) .."),
        ("a b c d e -> a b (c d) e", ".. c d e -> .. (c d) e"),
        ("a b c d e -> (a b c d e)", ".. -> (..)"),
        ("a b c d e -> b (c d e) a", "a b .. -> b (..) a"),
        ("a b c d e -> b (a c d) e", "a b .. e -> b (a ..) e"),
        input
    ];
    Ok(())
}

#[test]
fn equivalent_reduction() -> Result<()> {
    let input =
        Tensor::arange(0f32, (2 * 3 * 4 * 5 * 6) as f32, &Device::Cpu)?.reshape(&[2, 3, 4, 5, 6]);
    test![
        ("sum(a b c d e) -> ", "sum(..) -> "),
        ("a max(b c d) e -> (e a)", "a max(..) e -> (e a)"),
        (
            "a mean(b c) d e -> d (a e)",
            "a mean(b c) d e .. -> d (a e) .."
        ),
        ("a b min(c d e) -> (a b)", ".. min(c d e) -> (..)"),
        input
    ];

    Ok(())
}

macro_rules! seq_test {
    ($pattern1:literal, $pattern2:literal, $tensor:ident) => {
        let intermediate = einops!($pattern1, $tensor.clone());
        let output = einops!($pattern2, &intermediate.clone());
        assert_eq!(
            $tensor.clone().flatten_all()?.to_vec1::<f32>()?,
            output.flatten_all()?.to_vec1::<f32>()?, "({}) & ({}) failed", $pattern1, $pattern2
        );
    };
    ($(($pattern1:literal, $pattern2:literal)),*, $tensor:ident) => {
        $(seq_test!($pattern1, $pattern2, $tensor);)*
    };
}

#[test]
fn equivalent_repeat() -> Result<()> {
    let input = Tensor::arange(0f32, (1 * 2 * 4 * 6) as f32, &Device::Cpu)?.reshape(&[1, 2, 4, 6]);
    seq_test![
        (
            "a b c d -> (c 2 d a b)",
            "(c:4 min(2) d:6 a:1 b) -> a b c d"
        ),
        (
            "1 b c d -> (d copy:3 1) 3 b c",
            "(d min(copy:3) one:1) max(three:3) b c -> one b c d"
        ),
        (
            "1 .. d -> 1 (copy1:2 d copy2:3) ..",
            "1 (min(copy1:2) d max(copy2:3)) .. -> 1 .. d"
        ),
        ("1 .. -> 3 ..", "max(copy:3) .. -> 1 .."),
        (
            "1 b c d -> (1 1) (1 b) 2 c 3 d (1 1)",
            "(max(one1:1 one2:1)) (min(one3:1) b) min(two:2) c max(three:3) d (min(one4:1 one5:1)) -> 1 b c d"
        ),
        input
    ];

    let input = Tensor::arange(0f32, (2 * 3 * 5) as f32, &Device::Cpu)?.reshape(&[2, 3, 5]);
    seq_test![
        ("a b c -> c a b", "c a b -> a b c"),
        (
            "a b c -> (c copy:2 a b)",
            "(c:5 min(copy:2) a:2 b:3) -> a b c"
        ),
        ("a b c -> (a copy:1) b c", "(a min(copy:1)) b c -> a b c"),
        (
            "a b c -> (c a) (copy1:1 b copy2:2)",
            "(c a:2) (min(copy1:1) b max(copy2:2)) -> a b c"
        ),
        ("a .. -> a .. copy:4", "a .. max(copy:4) -> a .."),
        (
            ".. c -> .. (copy1:1 c copy2:2)",
            ".. (min(copy1:1) c max(copy2:2)) -> .. c"
        ),
        (
            ".. -> copy1:2 .. copy2:3",
            "max(copy1:2) .. min(copy2:3) -> .."
        ),
        (
            "a b c -> copy1:2 a copy2:1 b c 1",
            "min(copy1) a max(copy2) b c 1 -> a b c"
        ),
        input
    ];

    Ok(())
}

macro_rules! shape_test {
    ($pattern:literal, $shape:expr, $tensor:ident) => {
        let output = einops!($pattern, &$tensor);
        assert_eq!(output.shape(), $shape, "({}) pattern failed", $pattern);
    };
    ($(($pattern:literal, $shape:expr)),*, $tensor:ident) => {
        $(shape_test!($pattern, $shape, $tensor);)*
    };
}

#[test]
fn rearrange_reduce() -> Result<()> {
    let input =
        Tensor::arange(0f32, (10 * 20 * 30 * 40) as f32, &Device::Cpu)?.reshape(&[10, 20, 30, 40]);
    shape_test![
        ("b c h w -> b h w c", [10, 30, 40, 20]),
        ("b c h w -> b (c h w)", [10, 20 * 30 * 40]),
        (
            "b (c h1:2 w1:2) h w -> b c (h h1) (w w1)",
            [10, 5, 30 * 2, 40 * 2]
        ),
        (
            "b c (h h1:2) (w w1:2) -> b (h1 w1 c) h w",
            [10, 20 * 4, 30 / 2, 40 / 2]
        ),
        ("b1 sound b2 letter -> b1 b2 sound letter", [10, 30, 20, 40]),
        (
            "b c (h max(h1:2)) (w max(w1:2)) -> b c h w",
            [10, 20, 30 / 2, 40 / 2]
        ),
        ("b c max(h w) -> b c 1 1", [10, 20, 1, 1]),
        input
    ];

    Ok(())
}

#[test]
fn decomposition_variable() -> Result<()> {
    let input =
        Tensor::arange(0f32, (10 * 20 * 30 * 40) as f32, &Device::Cpu)?.reshape(&[10, 20, 30, 40]);

    let d2 = 2;
    let output1 = einops!("a b c (d1 {d2}) -> a b c d1 {d2}", &input);
    let output2 = einops!(".. (d1 {d2}) -> .. d1 {d2}", &input);
    assert_eq!(output1.shape(), &[10, 20, 30, 20, 2]);
    assert_eq!(output2.shape(), &[10, 20, 30, 20, 2]);

    let d2 = 2;
    let output1 = einops!("a b c (d1 sum({d2})) -> a b c d1", &input);
    let output2 = einops!(".. (d1 sum({d2})) -> .. d1", &input);
    assert_eq!(output1.shape(), &[10, 20, 30, 20]);
    assert_eq!(output2.shape(), &[10, 20, 30, 20]);

    let a1 = 5;
    let output = einops!("({a1} a2) .. -> a2 {a1} ..", &input);
    assert_eq!(output.shape(), &[2, 5, 20, 30, 40]);

    let shapes = (5, 2);
    let output = einops!(
        "({shapes.0} {shapes.1}) .. -> {shapes.1} {shapes.0} ..",
        &input
    );
    assert_eq!(output.shape(), &[2, 5, 20, 30, 40]);

    let shapes = (5, 2);
    let output = einops!(
        "({shapes.0} {shapes.1}) .. -> ({shapes.1} {shapes.0}) ..",
        &input
    );
    assert_eq!(output.shape(), &[10, 20, 30, 40]);

    Ok(())
}

#[test]
fn repeat_variable() -> Result<()> {
    let input =
        Tensor::arange(0f32, (10 * 20 * 30 * 40) as f32, &Device::Cpu)?.reshape(&[10, 20, 30, 40]);

    let repeat = 3;
    let output1 = einops!("a b c d -> {repeat} a b c d", &input);
    let output2 = einops!(".. -> {repeat} ..", &input);
    assert_eq!(output1.shape(), &[3, 10, 20, 30, 40]);
    assert_eq!(output2.shape(), &[3, 10, 20, 30, 40]);

    let repeat = 3;
    let output = einops!("a b c d -> ({repeat} a) b c d", &input);
    assert_eq!(output.shape(), &[30, 20, 30, 40]);

    Ok(())
}
