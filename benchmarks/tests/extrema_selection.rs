use candle_einops_benchmarks::extrema_spike::{ExtremaRoute, scenarios};

#[test]
fn extrema_spike_compares_selected_library_route_to_sequential_reference() {
    for scenario in scenarios() {
        assert_eq!(scenario.library_route(), ExtremaRoute::Selected);
        assert_eq!(scenario.reference_route(), ExtremaRoute::Sequential);
    }
}
