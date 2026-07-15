use candle_core::{Device, Result, Storage, Tensor, Var};
use candle_einops_benchmarks::permute_compose_layout_spike::{
    CollapseDecision, CopyReason, LayoutSpec, PublicFusionPlan, classify_permute_compose,
    classify_public_fusion, run_public_fusion_prototype,
};

fn layout(dims: &[usize], strides: &[usize], start_offset: usize) -> LayoutSpec {
    LayoutSpec::new(dims, strides, start_offset).expect("valid test layout")
}

fn storage_address(tensor: &Tensor) -> *const Storage {
    let (storage, _) = tensor.storage_and_layout();
    std::ptr::from_ref(&*storage)
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

#[test]
fn public_plan_reorders_only_whole_groups_and_has_exact_fallbacks() {
    let input = layout(&[2, 3, 4], &[12, 4, 1], 0);
    assert_eq!(
        classify_public_fusion(&input, &[vec![0], vec![1], vec![2]]).unwrap(),
        PublicFusionPlan::ExistingIdentityElision
    );
    assert!(matches!(
        classify_public_fusion(&input, &[vec![0], vec![1, 2]]).unwrap(),
        PublicFusionPlan::ExistingRouteView { .. }
    ));
    assert_eq!(
        classify_public_fusion(&input, &[vec![2], vec![0, 1]]).unwrap(),
        PublicFusionPlan::ReorderedPublicViews {
            pre_permutation: vec![0, 1, 2],
            reshape_dims: vec![6, 4],
            post_permutation: vec![1, 0],
        }
    );
    assert_eq!(
        classify_public_fusion(
            &layout(&[2, 3, 5, 7], &[105, 35, 7, 1], 0),
            &[vec![0], vec![2, 3], vec![1]],
        )
        .unwrap(),
        PublicFusionPlan::ReorderedPublicViews {
            pre_permutation: vec![0, 1, 2, 3],
            reshape_dims: vec![2, 3, 35],
            post_permutation: vec![0, 2, 1],
        }
    );
    assert!(matches!(
        classify_public_fusion(&input, &[vec![0, 2], vec![1]]).unwrap(),
        PublicFusionPlan::Fallback { .. }
    ));
    assert!(matches!(
        classify_public_fusion(
            &layout(&[2; 9], &[256, 128, 64, 32, 16, 8, 4, 2, 1], 0),
            &(0..9).rev().map(|axis| vec![axis]).collect::<Vec<_>>(),
        )
        .unwrap(),
        PublicFusionPlan::Fallback { .. }
    ));
}

#[test]
fn prototype_preserves_values_storage_layout_offsets_and_gradients() -> Result<()> {
    let device = Device::Cpu;
    let input = Var::from_vec(
        (0..24).map(|value| value as f32).collect(),
        (2, 3, 4),
        &device,
    )?;
    let old = input.permute([2, 0, 1])?.reshape(&[4, 6])?;
    let (candidate, plan) = run_public_fusion_prototype(input.as_tensor(), &[vec![2], vec![0, 1]])?;
    assert!(matches!(
        plan,
        PublicFusionPlan::ReorderedPublicViews { .. }
    ));
    assert_eq!(
        candidate.flatten_all()?.to_vec1::<f32>()?,
        old.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(candidate.dims(), [4, 6]);
    assert_eq!(candidate.stride(), [1, 4]);
    assert_eq!(
        storage_address(&candidate),
        storage_address(input.as_tensor())
    );
    assert_ne!(storage_address(&old), storage_address(input.as_tensor()));

    let weights = Tensor::arange(1f32, 25., &device)?.reshape(&[4, 6])?;
    let candidate_gradients = candidate.mul(&weights)?.sum_all()?.backward()?;
    let candidate_gradient = candidate_gradients
        .get(input.as_tensor())
        .expect("candidate must retain the original autograd edge");
    let old_gradients = old.mul(&weights)?.sum_all()?.backward()?;
    let old_gradient = old_gradients
        .get(input.as_tensor())
        .expect("existing route must retain the original autograd edge");
    assert_eq!(
        candidate_gradient.flatten_all()?.to_vec1::<f32>()?,
        old_gradient.flatten_all()?.to_vec1::<f32>()?
    );

    let storage = Tensor::arange(0f32, 48., &device)?.reshape(&[4, 3, 4])?;
    let offset = storage.narrow(0, 1, 2)?;
    let (offset_candidate, _) = run_public_fusion_prototype(&offset, &[vec![2], vec![0, 1]])?;
    assert_eq!(
        offset_candidate.layout().start_offset(),
        offset.layout().start_offset()
    );
    assert_eq!(storage_address(&offset_candidate), storage_address(&offset));
    Ok(())
}

#[test]
fn channel_flatten_singletons_zero_extents_and_fallback_match_existing_route() -> Result<()> {
    let device = Device::Cpu;
    let nchw = Tensor::arange(0f32, 2. * 3. * 5. * 7., &device)?.reshape(&[2, 3, 5, 7])?;
    let old = nchw.permute([0, 2, 3, 1])?.reshape(&[2, 35, 3])?;
    let (candidate, _) = run_public_fusion_prototype(&nchw, &[vec![0], vec![2, 3], vec![1]])?;
    assert_eq!(
        candidate.flatten_all()?.to_vec1::<f32>()?,
        old.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(candidate.stride(), [105, 1, 35]);
    assert_eq!(storage_address(&candidate), storage_address(&nchw));

    let singleton = Tensor::arange(0f32, 6., &device)?.reshape(&[2, 1, 3])?;
    let singleton_old = singleton.permute([1, 0, 2])?.reshape(&[1, 6])?;
    let (singleton_candidate, singleton_plan) =
        run_public_fusion_prototype(&singleton, &[vec![1], vec![0, 2]])?;
    assert!(matches!(
        singleton_plan,
        PublicFusionPlan::ExistingRouteView { .. }
    ));
    assert_eq!(singleton_candidate.layout(), singleton_old.layout());

    let zero = Tensor::zeros(&[2, 0, 3], candle_core::DType::F32, &device)?;
    let zero_old = zero.permute([2, 0, 1])?.reshape(&[3, 0])?;
    let (zero_candidate, zero_plan) = run_public_fusion_prototype(&zero, &[vec![2], vec![0, 1]])?;
    assert!(matches!(
        zero_plan,
        PublicFusionPlan::ReorderedPublicViews { .. }
    ));
    assert_eq!(zero_candidate.dims(), zero_old.dims());
    assert_eq!(storage_address(&zero_candidate), storage_address(&zero));

    let fallback_old = nchw.permute([0, 2, 1, 3])?.reshape(&[10, 3, 7])?;
    let (fallback, fallback_plan) =
        run_public_fusion_prototype(&nchw, &[vec![0, 2], vec![1], vec![3]])?;
    assert!(matches!(fallback_plan, PublicFusionPlan::Fallback { .. }));
    assert_eq!(
        fallback.flatten_all()?.to_vec1::<f32>()?,
        fallback_old.flatten_all()?.to_vec1::<f32>()?
    );
    assert_ne!(storage_address(&fallback), storage_address(&nchw));
    Ok(())
}
