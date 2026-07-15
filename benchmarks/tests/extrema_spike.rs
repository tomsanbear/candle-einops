use candle_core::{Device, Result};
use candle_einops_benchmarks::extrema_spike::{
    ExtremaKind, ExtremaLayout, collapsed_candidate, fixture, sequential, structural_metrics,
};

#[test]
fn candidate_matrix_has_one_reduction_and_only_strided_copy_pressure() -> Result<()> {
    for layout in [
        ExtremaLayout::ContiguousTrailing,
        ExtremaLayout::ContiguousLeading,
        ExtremaLayout::StridedTrailing,
    ] {
        let metrics = structural_metrics(layout);
        assert_eq!(metrics.current_submissions, 2);
        assert_eq!(metrics.candidate_submissions, 1);
        assert_eq!(
            metrics.candidate_materialized_elements > 0,
            layout == ExtremaLayout::StridedTrailing
        );
        let input = fixture(layout, &Device::Cpu)?;
        for kind in [ExtremaKind::Min, ExtremaKind::Max] {
            let current = sequential(&input, layout, kind)?;
            let candidate = collapsed_candidate(&input, layout, kind)?;
            assert_eq!(current.dims(), candidate.dims());
            assert_eq!(
                current.flatten_all()?.to_vec1::<f32>()?,
                candidate.flatten_all()?.to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}
