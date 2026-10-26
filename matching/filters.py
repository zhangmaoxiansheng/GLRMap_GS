"""One filter contract shared by matching, Track building and triangulation."""
import numpy as np

def shared_filter_settings(manifest):
    """Legacy defaults; experiment overrides apply consistently across stages."""
    settings=dict(match_min_overlap=.5,match_max_cycle_px=2.,match_max_epipolar_px=2.,
        multiview_min_angle_deg=2.,multiview_max_reprojection_px=2.,
        two_view_min_angle_deg=2.,two_view_min_overlap=.6,two_view_max_cycle_px=1.,
        two_view_max_sigma_px=8.,two_view_max_epipolar_px=1.5,two_view_max_reprojection_px=2.)
    overrides=manifest.get('filter_settings',{})
    if set(overrides)-set(settings):raise ValueError('Unknown shared filter setting')
    settings.update(overrides)
    if not all(np.isfinite(v) and v>0 for v in settings.values()):raise ValueError('Filters must be finite and positive')
    if not 0<settings['match_min_overlap']<=settings['two_view_min_overlap']<=1:raise ValueError('Invalid overlap thresholds')
    for two,multi in [('two_view_min_angle_deg','multiview_min_angle_deg')]:
        if settings[two]<settings[multi]:raise ValueError('Two-view angle must be at least the multiview angle')
    for two,multi in [('two_view_max_cycle_px','match_max_cycle_px'),('two_view_max_epipolar_px','match_max_epipolar_px'),('two_view_max_reprojection_px','multiview_max_reprojection_px')]:
        if settings[two]>settings[multi]:raise ValueError('Two-view error thresholds must be no looser than multiview thresholds')
    return settings
