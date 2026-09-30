"""Validated, optional native phase measurements appended to DONE."""
import math

FIELDS = {
    'cache_select_ms':'selection_ms',
    'cache_park_ms':'parking_ms',
    'cache_read_ms':'snapshot_read_ms',
    'cache_restore_ms':'snapshot_restore_ms',
    'checkpoint_ms':'checkpoint_ms',
    'prefill_replay_ms':'prefill_replay_ms',
    'prefill_new_ms':'prefill_new_ms',
    'prefill_mixed_ms':'prefill_mixed_ms',
    'first_generated_ms':'first_generated_ms',
}

def parse_phase_timings(parts):
    result = {}
    for part in parts:
        name, separator, value = part.partition('=')
        if not separator or name not in FIELDS:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number) and number >= 0:
            result[FIELDS[name]] = number
    return result
