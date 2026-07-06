"""Video LUT Generator — Professional LUT generation from reference video color styles."""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path


LOG_FILE = Path(__file__).parent / "lut_generator_debug.log"


def setup_logging() -> None:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[console_handler, file_handler],
    )


def global_excepthook(exc_type, exc_value, exc_tb) -> None:
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logging.critical("Unhandled exception:\n%s", msg)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n=== CRASH at {datetime.now()} ===\n{msg}\n")
    except Exception:
        pass


sys.excepthook = global_excepthook


def main() -> None:
    setup_logging()
    logging.info("=== Video LUT Generator starting ===")

    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Video LUT Generator")
    app.setOrganizationName("VideoLUT")
    window = MainWindow()
    window.show()
    logging.info("Main window displayed, entering event loop")
    exit_code = app.exec()
    logging.info("Application exiting with code %d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
