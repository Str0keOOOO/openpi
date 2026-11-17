"""Version and numerical settings of the current production dense teacher."""
VERSION = 'roma_ee_dense_v5_official_fast_single'
ROMA_SETTING = 'fast'
BIDIRECTIONAL = False
CONFIDENCE_THRESHOLD = None
FB_THRESHOLD_PX = None


def validate_model_info(info):
    expected = {
        'setting': ROMA_SETTING,
        'bidirectional': BIDIRECTIONAL,
        'input_resolution': [256, 256],
        'internal_lr_resolution': [512, 512],
        'internal_hr_resolution': [None, None],
        'inference_implementation': 'official_model_match',
    }
    for name, value in expected.items():
        if info.get(name) != value:
            raise ValueError(f'Incorrect fast teacher {name}: {info.get(name)!r}; expected {value!r}')


def validate_metadata(meta):
    if meta.get('cache_version') != VERSION or meta.get('roma_setting') != ROMA_SETTING or meta.get('roma_bidirectional') is not BIDIRECTIONAL:
        raise ValueError('Dense teacher must use the official Fast single-direction preset')
    if meta.get('fb_threshold_px') is not None or meta.get('confidence_threshold') is not None:
        raise ValueError('Official Fast cache cannot include extra confidence/cycle thresholds')
    validate_model_info(meta['model_info'])
