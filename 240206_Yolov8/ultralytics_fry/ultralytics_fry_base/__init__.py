# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.11"

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import RTDETR, SAM, YOLO
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

# _240319_2311_ fry修改
# from ultralytics_fry.ultralytics_fry_base.data.explorer.explorer import Explorer
# from ultralytics_fry.ultralytics_fry_base.models import RTDETR, SAM, YOLO
# from ultralytics_fry.ultralytics_fry_base.models.fastsam import FastSAM
# from ultralytics_fry.ultralytics_fry_base.models.nas import NAS
# from ultralytics_fry.ultralytics_fry_base.utils import ASSETS, SETTINGS as settings
# from ultralytics_fry.ultralytics_fry_base.utils.checks import check_yolo as checks
# from ultralytics_fry.ultralytics_fry_base.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
