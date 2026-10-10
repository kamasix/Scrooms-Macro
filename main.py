import sys

from shrooms_macro.app import main

if __name__ == "__main__":
    if "--self-check" in sys.argv:
        import json
        import cv2
        import numpy as np
        from shrooms_macro.config import Profile, application_dir
        from shrooms_macro.vision import Detector
        from shrooms_macro.platform_win import DPI_AWARENESS
        from shrooms_macro import __version__
        profile = Profile()
        detector = Detector(profile.data, profile.templates)
        result = {"version": __version__, "control_mode": profile.data["control_mode"],
                  "charge_action": profile.data["charge_action"],
                  "missing": detector.ready(), "opencv": cv2.__version__,
                  "numpy": np.__version__, "dpi": DPI_AWARENESS,
                  "source": "packaged assets only; no desktop capture or input"}
        (application_dir() / "self-check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        sys.exit(1 if result["missing"] else 0)
    else:
        main()
