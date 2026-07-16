use std::panic::{AssertUnwindSafe, catch_unwind};

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

use super::{Expression, ParsedExpression, private_ident};

const REGRESSION_SEEDS: &[&str] = &[
    // Missing delimiters and the parser unwraps removed during the original port.
    "a b",
    "a -> a ..",
    "a -> a b",
    "a b -> a",
    "() a -> a",
    "a -> () a",
    "sum() a -> a",
    // Grouped RHS errors that were previously overwritten by a later success.
    "a -> (a b 1)",
    "a -> (a a 1)",
    "a -> (a b)",
    // Duplicate and reduction identities formerly filtered before validation.
    "a b c -> b b (a)",
    "a .. -> a .. ..",
    "sum(axis axis) ->",
    "sum(.. ..) ->",
    "axis -> copy:2 copy:3 axis",
    "axis -> {copies} {copies} axis",
    "axis -> copies:2 {copies} axis",
    // Unsupported annotations and checked decomposition arithmetic.
    "a:3 b -> b a",
    "a b -> b a:3",
    "sum(axis:2) ->",
    "sum({axis}) ->",
    "(axis huge:18446744073709551615 two:2) -> axis huge two",
    "(axis zero:0) -> axis zero",
    // Runtime-rank and expansion-hygiene regressions still exercise planning.
    "a b .. -> .. b a",
    "rows columns -> rows columns {input}",
    "rows columns -> rows columns {input_shape}",
    "rows .. -> rows .. {input_ignored_len}",
    "sum(..) ->",
];

#[test]
fn historical_regressions_never_unwind() {
    assert_corpus(REGRESSION_SEEDS.iter().copied());
}

#[test]
fn bounded_arbitrary_utf8_never_unwinds() {
    let mut random = DeterministicRandom::new(0x7a62_15d4_d3c9_4f01);
    let corpus = (0..512)
        .map(|_| random.unicode_string(32))
        .collect::<Vec<_>>();
    assert_corpus(corpus.iter().map(String::as_str));
}

#[test]
fn bounded_grammar_aware_corpus_never_unwinds() {
    let mut random = DeterministicRandom::new(0x19e3_caf0_5eed_2026);
    let corpus = (0..1024)
        .map(|_| grammar_pattern(&mut random))
        .collect::<Vec<_>>();
    assert_corpus(corpus.iter().map(String::as_str));
}

fn assert_corpus<'a>(patterns: impl IntoIterator<Item = &'a str>) {
    for pattern in patterns {
        if planning_unwinds(pattern) {
            let minimized = minimize_unwind(pattern);
            panic!(
                "parser/token planning unwound; add this minimized input to REGRESSION_SEEDS: {minimized:?} (original {pattern:?})"
            );
        }
    }
}

fn planning_unwinds(pattern: &str) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let Ok(expression) = syn::parse_str::<Expression>(pattern) else {
            return;
        };

        let tensor = private_ident("property_input");
        let tensor_expression = quote!(let #tensor = (););
        let parsed = ParsedExpression {
            runtime_crate: syn::parse_quote!(::candle_einops),
            candle_crate: syn::parse_quote!(::candle_core),
            tensor,
            tensor_expression,
            expression,
        };
        let mut planned = TokenStream::new();
        parsed.to_tokens(&mut planned);
    }))
    .is_err()
}

fn minimize_unwind(pattern: &str) -> String {
    let mut current = pattern.chars().collect::<Vec<_>>();
    let mut index = 0;
    while index < current.len() {
        let mut candidate = current.clone();
        candidate.remove(index);
        let candidate = candidate.into_iter().collect::<String>();
        if planning_unwinds(&candidate) {
            current = candidate.chars().collect();
            index = 0;
        } else {
            index += 1;
        }
    }
    current.into_iter().collect()
}

fn grammar_pattern(random: &mut DeterministicRandom) -> String {
    const LEFT: &[&str] = &[
        "a",
        "b",
        "λ",
        "r#type",
        "..",
        "1",
        "2",
        "a:2",
        "a:0",
        "a:18446744073709551615",
        "{size}",
        "{config.width}",
        "{}",
        "(a b:2)",
        "(a:2 b:3)",
        "(a b)",
        "()",
        "sum(a)",
        "sum(..)",
        "sum(a:2)",
        "sum()",
        "#",
        ",",
    ];
    const RIGHT: &[&str] = &[
        "a",
        "b",
        "λ",
        "r#type",
        "..",
        "1",
        "2",
        "new:2",
        "new:0",
        "{size}",
        "{config.width}",
        "{}",
        "(a b)",
        "(a new:2 1)",
        "()",
        "sum(a)",
        "a:2",
        "#",
        ",",
    ];
    const ARROWS: &[&str] = &["->", "->", "->", "- >", "", "-> ->"];

    let left = random.sequence(LEFT, 6);
    let right = random.sequence(RIGHT, 6);
    format!("{left} {} {right}", random.choose(ARROWS))
}

struct DeterministicRandom(u64);

impl DeterministicRandom {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn choose<'a>(&mut self, choices: &'a [&str]) -> &'a str {
        choices[self.next() as usize % choices.len()]
    }

    fn sequence(&mut self, choices: &[&str], maximum_len: usize) -> String {
        let len = self.next() as usize % (maximum_len + 1);
        (0..len)
            .map(|_| self.choose(choices))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn unicode_string(&mut self, maximum_len: usize) -> String {
        let len = self.next() as usize % (maximum_len + 1);
        (0..len)
            .map(|_| {
                let scalar = (self.next() % 0x11_0000) as u32;
                char::from_u32(scalar).unwrap_or('\u{fffd}')
            })
            .collect()
    }
}
