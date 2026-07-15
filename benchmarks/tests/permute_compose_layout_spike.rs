use candle_einops_benchmarks::permute_compose_layout_spike::{
    CollapseDecision, CopyReason, LayoutSpec, classify_permute_compose,
};

fn layout(dims: &[usize], strides: &[usize], start_offset: usize) -> LayoutSpec {
    LayoutSpec::new(dims, strides, start_offset).expect("valid test layout")
}

#[test]
fn decision_table_distinguishes_existing_views_layout_candidates_and_copies() {
    let cases = [
        (
            "identity-is-owned-by-existing-elision",
            layout(&[2, 3, 4], &[12, 4, 1], 0),
            vec![0, 1, 2],
            vec![vec![0], vec![1], vec![2]],
            CollapseDecision::ExistingIdentityElision,
        ),
        (
            "naturally-viewable-adjacent-group",
            layout(&[2, 3, 4], &[12, 4, 1], 0),
            vec![0, 1, 2],
            vec![vec![0], vec![1, 2]],
            CollapseDecision::ExistingPublicReshapeView {
                output_dims: vec![2, 12],
                output_strides: vec![12, 1],
                start_offset: 0,
            },
        ),
        (
            "c-ab-is-stride-collapsible-but-not-publicly-constructible",
            layout(&[2, 3, 4], &[12, 4, 1], 0),
            vec![2, 0, 1],
            vec![vec![0], vec![1, 2]],
            CollapseDecision::LayoutOnlyCandidate {
                output_dims: vec![4, 6],
                output_strides: vec![1, 4],
                start_offset: 0,
            },
        ),
        (
            "n-hw-c-is-stride-collapsible-but-not-publicly-constructible",
            layout(&[2, 3, 5, 7], &[105, 35, 7, 1], 0),
            vec![0, 2, 3, 1],
            vec![vec![0], vec![1, 2], vec![3]],
            CollapseDecision::LayoutOnlyCandidate {
                output_dims: vec![2, 35, 3],
                output_strides: vec![105, 1, 35],
                start_offset: 0,
            },
        ),
        (
            "c-b-cannot-collapse-in-logical-order",
            layout(&[2, 3, 4], &[12, 4, 1], 0),
            vec![2, 1, 0],
            vec![vec![0, 1], vec![2]],
            CollapseDecision::CopyRequired(CopyReason::NonCollapsibleBoundary {
                outer_axis: 0,
                inner_axis: 1,
            }),
        ),
    ];

    for (name, input, permutation, groups, expected) in cases {
        assert_eq!(
            classify_permute_compose(&input, &permutation, &groups).unwrap(),
            expected,
            "{name}"
        );
    }
}

#[test]
fn offsets_singletons_zero_extents_and_invalid_metadata_are_conservative() {
    assert_eq!(
        classify_permute_compose(
            &layout(&[2, 3, 4], &[12, 4, 1], 24),
            &[2, 0, 1],
            &[vec![0], vec![1, 2]],
        )
        .unwrap(),
        CollapseDecision::LayoutOnlyCandidate {
            output_dims: vec![4, 6],
            output_strides: vec![1, 4],
            start_offset: 24,
        }
    );

    assert_eq!(
        classify_permute_compose(
            &layout(&[2, 1, 3], &[3, 91, 1], 0),
            &[1, 0, 2],
            &[vec![0, 1, 2]],
        )
        .unwrap(),
        CollapseDecision::ExistingPublicReshapeView {
            output_dims: vec![6],
            output_strides: vec![1],
            start_offset: 0,
        }
    );

    assert_eq!(
        classify_permute_compose(
            &layout(&[2, 0, 3], &[0, 3, 1], 0),
            &[0, 1, 2],
            &[vec![0], vec![1, 2]],
        )
        .unwrap(),
        CollapseDecision::CopyRequired(CopyReason::ZeroExtentConservative)
    );

    assert!(
        classify_permute_compose(
            &layout(&[2, 3, 4], &[12, 4, 1], 0),
            &[0, 0, 2],
            &[vec![0], vec![1, 2]],
        )
        .is_err()
    );
    assert!(
        classify_permute_compose(
            &layout(&[2, 3, 4], &[12, 4, 1], 0),
            &[0, 1, 2],
            &[vec![0], vec![2]],
        )
        .is_err()
    );
}
