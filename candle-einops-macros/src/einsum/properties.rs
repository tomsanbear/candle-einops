use std::panic::{AssertUnwindSafe, catch_unwind};

use proc_macro2::Span;
use quote::quote;

use super::{Equation, Invocation};

const REGRESSION_SEEDS: &[&str] = &[
    "",
    "->",
    "a",
    "a ->",
    "-> a",
    "a -> a",
    "a a -> a",
    "a, a -> a",
    ".. a, a .. -> .. a",
    ".. .. -> ..",
    "a -> a a",
    "a -> missing",
    "a-b -> a-b",
    "λ β, β γ -> λ γ",
    "a, b, c ->",
    "a -> a -> a",
    "a -> a, b",
];

#[test]
fn historical_einsum_regressions_never_unwind() {
    assert_corpus(REGRESSION_SEEDS.iter().copied());
}

#[test]
fn bounded_arbitrary_utf8_einsum_never_unwinds() {
    let mut random = DeterministicRandom::new(0xe175_0a2d_2026_0715);
    let corpus = (0..512)
        .map(|_| random.unicode_string(48))
        .collect::<Vec<_>>();
    assert_corpus(corpus.iter().map(String::as_str));
}

#[test]
fn bounded_grammar_aware_einsum_ir_never_unwinds() {
    let mut random = DeterministicRandom::new(0x1a81_5f17_cafe_0b1e);
    let corpus = (0..1024)
        .map(|_| grammar_equation(&mut random))
        .collect::<Vec<_>>();
    assert_corpus(corpus.iter().map(String::as_str));
}

fn assert_corpus<'a>(equations: impl IntoIterator<Item = &'a str>) {
    for equation in equations {
        if planning_unwinds(equation) {
            let minimized = minimize_unwind(equation);
            panic!(
                "einsum parser/IR unwound; add this minimized input to REGRESSION_SEEDS: {minimized:?} (original {equation:?})"
            );
        }
    }
}

fn planning_unwinds(text: &str) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let literal = syn::LitStr::new(text, Span::call_site());
        if let Ok(equation) = Equation::parse(&literal) {
            if equation.operands.len() == 1 && !equation.requires_runtime_normalization() {
                let _ = equation.unary_permutation();
            } else if equation.operands.len() == 2 && !equation.requires_runtime_normalization() {
                let _ = equation.binary_plan();
            }
        }

        for operand_count in 0..=4 {
            let operands = (0..operand_count).map(|_| quote!(()));
            let invocation = quote!(#literal, #(#operands),*);
            let _ = syn::parse2::<Invocation>(invocation);
        }
    }))
    .is_err()
}

fn minimize_unwind(equation: &str) -> String {
    let mut current = equation.chars().collect::<Vec<_>>();
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

fn grammar_equation(random: &mut DeterministicRandom) -> String {
    const LABELS: &[&str] = &[
        "a", "b", "row", "inner", "λ", "_axis", "..", "...", "1", "a-b", "a:b", "#",
    ];
    const ARROWS: &[&str] = &["->", "->", "->", "", "- >", "-> ->"];

    let operand_count = random.next() as usize % 6;
    let inputs = (0..operand_count)
        .map(|_| random.sequence(LABELS, 7))
        .collect::<Vec<_>>()
        .join(", ");
    let output = random.sequence(LABELS, 6);
    format!("{inputs} {} {output}", random.choose(ARROWS))
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
        let length = self.next() as usize % (maximum_len + 1);
        (0..length)
            .map(|_| self.choose(choices))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn unicode_string(&mut self, maximum_len: usize) -> String {
        let length = self.next() as usize % (maximum_len + 1);
        (0..length)
            .map(|_| {
                let scalar = (self.next() % 0x11_0000) as u32;
                char::from_u32(scalar).unwrap_or('\u{fffd}')
            })
            .collect()
    }
}
