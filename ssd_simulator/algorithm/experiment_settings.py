"""Use the saved settings and apply the loss override from the recorded runner"""

import json
from pathlib import Path

HOUR = 3600000000
TRAINING = [dict(id="1h_all_uniform", refresh_hours=1, window_hours=0, sample_weight="uniform")]
RECORDED = json.loads((Path(__file__).with_name("recorded_settings.json")).read_text())
SETTINGS = dict(RECORDED, objective="raw_l2")


def validate_recorded_settings(settings):
    """Reject changes to the selected eight-input recipe."""
    if settings != SETTINGS:
        raise ValueError("Expected the recorded raw-target SSD recipe")
    return dict(settings)
