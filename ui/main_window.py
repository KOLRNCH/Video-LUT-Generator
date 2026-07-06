import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPixmap,
    QFont,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from core.video_loader import VideoLoader
from core.frame_extractor import FrameExtractor
from core.color_analysis import ColorAnalyzer, ColorStats
from core.lut_generator import LUTGenerator
from core.cube_exporter import CubeExporter
from ui.video_preview import LUTVideoPlayer, ComparisonDialog

logger = logging.getLogger(__name__)

DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #0d0d0d;
    color: #f5f5f5;
}
QLabel {
    color: #f5f5f5;
    background-color: transparent;
}
QMenuBar {
    background-color: #0d0d0d;
    color: #ccc;
    border-bottom: 1px solid #1a1a1a;
    padding: 2px;
}
QMenuBar::item:selected {
    background-color: #7C5CFC;
    color: white;
    border-radius: 4px;
}
QMenu {
    background-color: #1a1a1a;
    color: #f5f5f5;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 6px;
}
QMenu::item {
    padding: 8px 28px 8px 16px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #7C5CFC;
}
QMenu::separator {
    height: 1px;
    background-color: #2a2a2a;
    margin: 4px 8px;
}
QPushButton {
    background-color: #7C5CFC;
    color: white;
    border: none;
    padding: 10px 22px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #6B4FE0;
}
QPushButton:pressed {
    background-color: #5A42C4;
}
QPushButton:disabled {
    background-color: #1a1a1a;
    color: #444;
    border: 1px solid #2a2a2a;
}
QPushButton#successBtn {
    background-color: #10B981;
    font-weight: 700;
    padding: 12px 28px;
    font-size: 14px;
}
QPushButton#successBtn:hover {
    background-color: #059669;
}
QPushButton#dangerBtn {
    background-color: #EF4444;
}
QPushButton#dangerBtn:hover {
    background-color: #DC2626;
}
QPushButton#saveBtn {
    background-color: transparent;
    border: 1.5px solid #7C5CFC;
    color: #7C5CFC;
    padding: 8px 18px;
}
QPushButton#saveBtn:hover {
    background-color: #7C5CFC;
    color: white;
}
QPushButton#secondaryBtn {
    background-color: #1a1a1a;
    border: 1px solid #333;
    color: #ccc;
}
QPushButton#secondaryBtn:hover {
    background-color: #2a2a2a;
    border-color: #7C5CFC;
    color: #fff;
}
QProgressBar {
    border: none;
    border-radius: 6px;
    text-align: center;
    background-color: #1a1a1a;
    color: #f5f5f5;
    height: 22px;
    font-size: 11px;
    font-weight: 500;
}
QProgressBar::chunk {
    background-color: #7C5CFC;
    border-radius: 6px;
}
QComboBox {
    background-color: #1a1a1a;
    color: #f5f5f5;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox::down-arrow {
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #7C5CFC;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1a1a1a;
    color: #f5f5f5;
    selection-background-color: #7C5CFC;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #2a2a2a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #7C5CFC;
    border: none;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #7C5CFC;
    border-radius: 2px;
}
QFrame#dropArea {
    background-color: #1a1a1a;
    border: 2px dashed #333;
    border-radius: 16px;
}
QFrame#dropArea:hover {
    border-color: #7C5CFC;
    background-color: #1e1e2e;
}
QFrame#dropAreaActive {
    background-color: #1e1e2e;
    border: 2px solid #7C5CFC;
    border-radius: 16px;
}
QFrame#cardDivider {
    background-color: transparent;
    border: none;
    border-top: 1px solid #1a1a1a;
}
"""


class VideoPanel(QFrame):
    """Premium drag-and-drop panel for video selection."""

    fileDropped = pyqtSignal(str)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("dropArea")
        self.setAcceptDrops(True)
        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._title = title
        self._file_path: Optional[str] = None
        self._pixmap: Optional[QPixmap] = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        self._icon_label = QLabel("\u25C9")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setFont(QFont("Segoe UI", 28))
        self._icon_label.setStyleSheet("color: #555;")
        layout.addWidget(self._icon_label)

        self._title_label = QLabel(f"<b>{self._title}</b>")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setFont(QFont("Segoe UI", 13))
        self._title_label.setStyleSheet("color: #ccc;")
        layout.addWidget(self._title_label)

        self._hint_label = QLabel("Drag & drop video here\nor click to browse")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setFont(QFont("Segoe UI", 10))
        self._hint_label.setStyleSheet("color: #555;")
        layout.addWidget(self._hint_label)

        self._path_label = QLabel("")
        self._path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._path_label.setFont(QFont("Segoe UI", 9))
        self._path_label.setStyleSheet("color: #888;")
        self._path_label.setWordWrap(True)
        self._path_label.setVisible(False)
        layout.addWidget(self._path_label)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setVisible(False)
        layout.addWidget(self._preview_label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setObjectName("dropAreaActive")
            self.setStyle(self.style())

    def dragLeaveEvent(self, event) -> None:
        self.setObjectName("dropArea")
        self.setStyle(self.style())

    def dropEvent(self, event: QDropEvent) -> None:
        self.setObjectName("dropArea")
        self.setStyle(self.style())
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv")):
                self.set_file(path)
                self.fileDropped.emit(path)
            else:
                QMessageBox.warning(self, "Unsupported Format", "Please drop a video file.")

    def mousePressEvent(self, event) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self._title}",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)",
        )
        if path:
            self.set_file(path)
            self.fileDropped.emit(path)

    def set_file(self, path: str) -> None:
        self._file_path = path
        name = Path(path).name
        self._path_label.setText(f"\u25B6 {name}")
        self._path_label.setVisible(True)
        self._hint_label.setVisible(False)
        self._title_label.setVisible(False)
        self._icon_label.setVisible(False)
        self._show_preview(path)

    def _show_preview(self, path: str) -> None:
        try:
            loader = VideoLoader(path)
            frame = loader.read_frame_at(loader.frame_count // 2)
            loader.release()
            if frame is not None:
                h, w = frame.shape[:2]
                aspect = w / h
                disp_h = 140
                disp_w = int(disp_h * aspect)
                resized = cv2.resize(frame, (disp_w, disp_h))
                h2, w2 = resized.shape[:2]
                img = QImage(resized.data.tobytes(), w2, h2, 3 * w2, QImage.Format.Format_RGB888)
                self._pixmap = QPixmap.fromImage(img)
                self._preview_label.setPixmap(self._pixmap)
                self._preview_label.setVisible(True)
        except Exception as e:
            logger.warning("Could not generate preview: %s", e)

    def file_path(self) -> Optional[str]:
        return self._file_path


class WorkerThread(QThread):
    """Performs video analysis and LUT generation in a background thread."""

    progress = pyqtSignal(int, str)
    generation_done = pyqtSignal(object)
    generation_error = pyqtSignal(str)

    def __init__(
        self,
        ref_path: str,
        tgt_path: str,
        lut_size: int,
        strength: float,
        mode: str = "natural",
    ) -> None:
        super().__init__()
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.lut_size = lut_size
        self.strength = strength
        self.mode = mode

    def run(self) -> None:
        try:
            self.progress.emit(5, "Loading reference video...")
            ref_loader = VideoLoader(self.ref_path)

            self.progress.emit(10, "Loading target video...")
            tgt_loader = VideoLoader(self.tgt_path)

            extractor = FrameExtractor(num_frames=75)

            self.progress.emit(15, "Extracting frames from reference...")
            ref_frames = extractor.extract(ref_loader)
            ref_loader.release()
            if not ref_frames:
                self.generation_error.emit("Could not extract frames from reference video.")
                return

            self.progress.emit(30, "Extracting frames from target...")
            tgt_frames = extractor.extract(tgt_loader)
            tgt_loader.release()
            if not tgt_frames:
                self.generation_error.emit("Could not extract frames from target video.")
                return

            analyzer = ColorAnalyzer()

            self.progress.emit(45, "Analyzing reference color palette...")
            ref_stats = analyzer.analyze(ref_frames)

            self.progress.emit(60, "Analyzing target color palette...")
            tgt_stats = analyzer.analyze(tgt_frames)

            self.progress.emit(75, "Generating LUT...")
            generator = LUTGenerator(size=self.lut_size)
            lut = generator.generate(ref_stats, tgt_stats, strength=self.strength, tgt_frames=tgt_frames, ref_frames=ref_frames, mode=self.mode)

            self.progress.emit(95, "Applying LUT to preview frame...")
            preview_loader = VideoLoader(self.tgt_path)
            preview_frame = preview_loader.read_frame_at(preview_loader.frame_count // 2)
            preview_loader.release()

            applied_frame = None
            if preview_frame is not None and generator.lut is not None:
                applied_frame = generator.apply_lut(preview_frame)

            result = {
                "ref_stats": ref_stats,
                "tgt_stats": tgt_stats,
                "lut": generator.lut,
                "original_frame": preview_frame,
                "applied_frame": applied_frame,
                "generator": generator,
            }

            self.progress.emit(100, "Done!")
            self.generation_done.emit(result)

        except Exception as e:
            logger.exception("Worker failed")
            self.generation_error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for Video LUT Generator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video LUT Generator")
        self.setMinimumSize(960, 680)
        self.setStyleSheet(DARK_STYLE)

        self._ref_stats: Optional[ColorStats] = None
        self._tgt_stats: Optional[ColorStats] = None
        self._lut: Optional[np.ndarray] = None
        self._original_frame: Optional[np.ndarray] = None
        self._applied_frame: Optional[np.ndarray] = None
        self._worker: Optional[WorkerThread] = None
        self._generated_luts: List[dict] = []
        self._target_path: Optional[str] = None
        self._lut_gen: Optional[LUTGenerator] = None

        self._init_menu()
        self._init_ui()

    def _init_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_ref = QAction("Open &Reference Video...", self)
        open_ref.triggered.connect(self._open_ref_dialog)
        file_menu.addAction(open_ref)

        open_tgt = QAction("Open &Target Video...", self)
        open_tgt.triggered.connect(self._open_tgt_dialog)
        file_menu.addAction(open_tgt)

        file_menu.addSeparator()

        save_action = QAction("&Save LUT...", self)
        save_action.triggered.connect(self._save_lut)
        save_action.setEnabled(False)
        self._save_action = save_action
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(24, 20, 24, 20)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)

        title = QLabel("<h1 style='font-weight: 300; letter-spacing: 1px;'>Video LUT Generator</h1>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 20))
        title.setStyleSheet("color: #f5f5f5;")
        header_layout.addWidget(title)

        subtitle = QLabel(
            "Capture the color style from a reference video and apply it to any footage."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #777; font-size: 13px;")
        header_layout.addWidget(subtitle)

        main_layout.addLayout(header_layout)

        panel_layout = QHBoxLayout()
        panel_layout.setSpacing(20)

        self._ref_panel = VideoPanel("Reference Video")
        self._ref_panel.fileDropped.connect(self._on_ref_dropped)
        self._tgt_panel = VideoPanel("Target Video")
        self._tgt_panel.fileDropped.connect(self._on_tgt_dropped)
        panel_layout.addWidget(self._ref_panel, stretch=1)
        panel_layout.addWidget(self._tgt_panel, stretch=1)
        main_layout.addLayout(panel_layout)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setValue(0)
        self._progress.setFixedHeight(24)
        main_layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        self._status_label.setWordWrap(True)
        main_layout.addWidget(self._status_label)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        controls_layout.addStretch()

        param_group = QHBoxLayout()
        param_group.setSpacing(6)

        lut_size_label = QLabel("Size")
        lut_size_label.setFont(QFont("Segoe UI", 10))
        lut_size_label.setStyleSheet("color: #999;")
        param_group.addWidget(lut_size_label)

        self._size_combo = QComboBox()
        self._size_combo.addItems(["17x17x17", "33x33x33", "65x65x65"])
        self._size_combo.setCurrentIndex(1)
        self._size_combo.setFont(QFont("Segoe UI", 11))
        self._size_combo.setFixedWidth(110)
        param_group.addWidget(self._size_combo)

        param_group.addSpacing(8)

        strength_label = QLabel("Strength")
        strength_label.setFont(QFont("Segoe UI", 10))
        strength_label.setStyleSheet("color: #999;")
        param_group.addWidget(strength_label)

        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(10, 100)
        self._strength_slider.setValue(80)
        self._strength_slider.setFixedWidth(100)
        param_group.addWidget(self._strength_slider)

        self._strength_val_label = QLabel("0.80")
        self._strength_val_label.setFont(QFont("Segoe UI", 11))
        self._strength_val_label.setStyleSheet("color: #7C5CFC; font-weight: 600;")
        self._strength_val_label.setFixedWidth(36)
        param_group.addWidget(self._strength_val_label)

        self._strength_slider.valueChanged.connect(
            lambda v: self._strength_val_label.setText(f"{v/100:.2f}")
        )

        param_group.addSpacing(8)

        mode_label = QLabel("Mode")
        mode_label.setFont(QFont("Segoe UI", 10))
        mode_label.setStyleSheet("color: #999;")
        param_group.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Natural", "Aggressive"])
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.setFont(QFont("Segoe UI", 11))
        self._mode_combo.setFixedWidth(110)
        param_group.addWidget(self._mode_combo)

        controls_layout.addLayout(param_group)

        controls_layout.addSpacing(16)

        self._generate_btn = QPushButton("Generate LUT")
        self._generate_btn.setObjectName("successBtn")
        self._generate_btn.setEnabled(False)
        self._generate_btn.clicked.connect(self._generate_lut)
        self._generate_btn.setFixedWidth(170)
        self._generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_layout.addWidget(self._generate_btn)

        controls_layout.addStretch()

        main_layout.addLayout(controls_layout)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)
        action_layout.addStretch()

        self._save_btn = QPushButton("Save LUT")
        self._save_btn.setObjectName("saveBtn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_lut)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        action_layout.addWidget(self._save_btn)

        self._result_btn = QPushButton("Preview Video")
        self._result_btn.setObjectName("secondaryBtn")
        self._result_btn.setEnabled(False)
        self._result_btn.clicked.connect(self._show_result)
        self._result_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        action_layout.addWidget(self._result_btn)

        self._compare_btn = QPushButton("Compare LUTs")
        self._compare_btn.setObjectName("secondaryBtn")
        self._compare_btn.setEnabled(False)
        self._compare_btn.clicked.connect(self._compare_luts)
        self._compare_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        action_layout.addWidget(self._compare_btn)

        action_layout.addStretch()

        main_layout.addLayout(action_layout)

    def _open_ref_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)"
        )
        if path:
            self._ref_panel.set_file(path)
            self._on_ref_dropped(path)

    def _open_tgt_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)"
        )
        if path:
            self._tgt_panel.set_file(path)
            self._on_tgt_dropped(path)

    def _on_ref_dropped(self, path: str) -> None:
        self._check_generate_ready()

    def _on_tgt_dropped(self, path: str) -> None:
        self._target_path = path
        self._check_generate_ready()

    def _check_generate_ready(self) -> None:
        ready = (
            self._ref_panel.file_path() is not None
            and self._tgt_panel.file_path() is not None
        )
        self._generate_btn.setEnabled(ready)

    def _generate_lut(self) -> None:
        ref_path = self._ref_panel.file_path()
        tgt_path = self._tgt_panel.file_path()
        if not ref_path or not tgt_path:
            return

        size_text = self._size_combo.currentText()
        size = int(size_text.split("x")[0])
        strength = self._strength_slider.value() / 100.0
        mode = self._mode_combo.currentText().lower()

        self._generate_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._result_btn.setEnabled(False)
        self._compare_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Initializing...")

        self._worker = WorkerThread(ref_path, tgt_path, size, strength, mode)
        self._worker.progress.connect(self._on_progress)
        self._worker.generation_done.connect(self._on_finished)
        self._worker.generation_error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        self._progress.setValue(value)
        self._status_label.setText(message)

    def _on_finished(self, result: dict) -> None:
        try:
            self._ref_stats = result.get("ref_stats")
            self._tgt_stats = result.get("tgt_stats")
            self._lut = result.get("lut")
            self._original_frame = result.get("original_frame")
            self._applied_frame = result.get("applied_frame")

            gen = result.get("generator")
            self._lut_gen = gen

            size_text = self._size_combo.currentText()
            size = int(size_text.split("x")[0])
            strength = self._strength_slider.value() / 100.0
            mode = self._mode_combo.currentText().lower()

            if self._lut is not None:
                self._generated_luts.append({
                    "lut": self._lut,
                    "params": {"size": size, "strength": strength, "mode": mode},
                })

            self._progress.setVisible(False)

            quality_text = ""
            if gen and hasattr(gen, "quality_text"):
                quality_text = gen.quality_text()

            if quality_text:
                self._status_label.setText(
                    f"LUT generated! {quality_text}"
                )
            else:
                self._status_label.setText(
                    "LUT generated successfully!"
                )
            self._generate_btn.setEnabled(True)
            self._save_btn.setEnabled(True)
            self._result_btn.setEnabled(True)
            self._compare_btn.setEnabled(len(self._generated_luts) >= 2)
        except Exception as e:
            logger.exception("Error in _on_finished: %s", e)
        finally:
            self._worker = None

    def _on_error(self, msg: str) -> None:
        try:
            self._progress.setVisible(False)
            self._status_label.setText("Error occurred")
            self._generate_btn.setEnabled(True)
            self._compare_btn.setEnabled(len(self._generated_luts) >= 2)
            QMessageBox.critical(self, "Error", f"An error occurred:\n\n{msg}")
        except Exception as e:
            logger.exception("Error in _on_error: %s", e)
        finally:
            self._worker = None

    def _save_lut(self) -> None:
        if self._lut is None:
            QMessageBox.warning(self, "No LUT", "Generate a LUT first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LUT as .cube",
            "color_lut.cube",
            "Cube LUT (*.cube);;All Files (*)",
        )
        if not path:
            return

        try:
            CubeExporter.export(self._lut, path, title="Video LUT Generator")
            ok = CubeExporter.validate(path)
            if ok:
                QMessageBox.information(
                    self, "Saved",
                    f"LUT saved to:\n{path}\n\n"
                    "Compatible with DaVinci Resolve, Premiere Pro, "
                    "After Effects, Final Cut Pro, OBS Studio.",
                )
            else:
                QMessageBox.warning(self, "Warning", "File saved but validation failed.")
        except Exception as e:
            logger.exception("Failed to save LUT")
            QMessageBox.critical(self, "Save Error", str(e))

    def _show_result(self) -> None:
        if self._target_path is None or self._lut is None:
            QMessageBox.warning(self, "No Preview", "Generate a LUT first.")
            return
        player = LUTVideoPlayer(
            video_path=self._target_path,
            lut=self._lut,
            title="",
            max_width=520,
        )

        class PreviewDialog(QDialog):
            def __init__(self, p: LUTVideoPlayer, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._player = p

            def closeEvent(self, event):
                self._player.shutdown()
                super().closeEvent(event)

        dialog = PreviewDialog(player, self)
        dialog.setWindowTitle("Video Preview \u2014 Original vs LUT")
        dialog.setMinimumSize(1000, 600)
        dialog.setStyleSheet(DARK_STYLE)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 16, 20, 16)
        title = QLabel("<h2 style='font-weight: 300;'>Video Preview</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(player)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("dangerBtn")
        close_btn.clicked.connect(dialog.close)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        dialog.exec()

    def _compare_luts(self) -> None:
        if self._target_path is None or not self._generated_luts:
            QMessageBox.warning(self, "No LUTs", "Generate at least one LUT first.")
            return
        if len(self._generated_luts) == 1:
            QMessageBox.information(
                self, "Only One LUT",
                "Generate more LUTs with different settings to compare them.\n\n"
                "Change LUT size, strength, or mode and generate again.",
            )
            return
        dialog = ComparisonDialog(
            video_path=self._target_path,
            luts=self._generated_luts,
            parent=self,
        )
        dialog.exec()

    def _show_about(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("About Video LUT Generator")
        dialog.setFixedSize(460, 560)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #f5f5f5;
            }
            QLabel {
                color: #f5f5f5;
                background-color: transparent;
            }
            QPushButton {
                background-color: #7C5CFC;
                color: white;
                border: none;
                padding: 10px 32px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #6B4FE0;
            }
        """)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_path = Path(__file__).parent.parent / "LOGOwhite.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            scaled = pix.scaled(200, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_lbl = QLabel()
            logo_lbl.setPixmap(scaled)
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_lbl)
            layout.addSpacing(8)

        title = QLabel("<h2 style='font-weight: 300; letter-spacing: 1px; color: #f5f5f5;'>Video LUT Generator</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        ver = QLabel("<span style='color: #7C5CFC; font-size: 13px;'>v1.0</span>")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)
        layout.addSpacing(12)

        desc = QLabel(
            "Generates professional .cube LUTs from reference video color palettes."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(desc)
        layout.addSpacing(16)

        algo_title = QLabel("<b style='color: #ccc;'>Algorithms</b>")
        algo_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(algo_title)

        algos = [
            "CIE LAB Color Space",
            "Reinhard Color Transfer",
            "Histogram Matching",
            "Tonal Separation",
        ]
        for a in algos:
            lbl = QLabel(f"\u2022  {a}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(lbl)

        layout.addSpacing(16)

        tech = QLabel("Built with Python, PyQt6, OpenCV, NumPy, Numba")
        tech.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tech.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(tech)

        layout.addSpacing(8)

        link = QLabel(
            '<a href="https://sup.kolrn.workers.dev/" style="color: #7C5CFC; text-decoration: none;">sup.kolrn.workers.dev</a>'
        )
        link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        link.setOpenExternalLinks(True)
        link.setStyleSheet("font-size: 12px;")
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(link)

        layout.addStretch()

        btn = QPushButton("Close")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

        dialog.exec()
