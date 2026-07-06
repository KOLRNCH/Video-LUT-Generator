import logging
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from core.cube_exporter import CubeExporter

logger = logging.getLogger(__name__)

DARK_STYLE = """
QMainWindow, QDialog, QScrollArea {
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
    background-color: #2a2a2a;
    color: #555;
}
QPushButton#saveBtn {
    background-color: transparent;
    border: 1.5px solid #7C5CFC;
    color: #7C5CFC;
    padding: 6px 14px;
    font-size: 11px;
    min-height: 16px;
}
QPushButton#saveBtn:hover {
    background-color: #7C5CFC;
    color: white;
}
QPushButton#playBtn {
    background-color: #10B981;
    border: none;
    padding: 6px 10px;
    font-size: 14px;
    min-height: 16px;
    border-radius: 6px;
}
QPushButton#playBtn:hover {
    background-color: #059669;
}
QPushButton#dangerBtn {
    background-color: #EF4444;
}
QPushButton#dangerBtn:hover {
    background-color: #DC2626;
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
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #7C5CFC;
    border-radius: 2px;
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
    image: none;
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
QProgressBar {
    border: none;
    border-radius: 6px;
    text-align: center;
    background-color: #1a1a1a;
    color: #f5f5f5;
    height: 20px;
    font-size: 11px;
    font-weight: 500;
}
QProgressBar::chunk {
    background-color: #7C5CFC;
    border-radius: 6px;
}
QScrollBar:vertical {
    background: #0d0d0d;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #333;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #555;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class LUTVideoPlayer(QWidget):
    """Plays a video with LUT applied in real-time with playback controls."""

    def __init__(
        self,
        video_path: str,
        lut: np.ndarray,
        title: str = "",
        max_width: int = 480,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._lut = lut
        self._title = title
        self._max_width = max_width
        self._cap: Optional[cv2.VideoCapture] = None
        self._playing = False
        self._current_frame_idx = 0
        self._total_frames = 0
        self._fps = 30.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._target_delay_ms = 33
        self._use_lut = True
        self._closed = False

        self.setStyleSheet(self._player_style())
        self._init_ui()
        self._open_video()

    def _player_style(self) -> str:
        return """
        QPushButton#playBtn {
            background-color: #10B981;
            border: none;
            padding: 4px 6px;
            font-size: 18px;
            font-family: 'Segoe UI', 'Arial Unicode MS', sans-serif;
            min-height: 18px;
            border-radius: 6px;
            color: white;
        }
        QPushButton#playBtn:hover {
            background-color: #059669;
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
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::sub-page:horizontal {
            background: #7C5CFC;
            border-radius: 2px;
        }
        QLabel {
            color: #d4d4d4;
            background-color: transparent;
        }
        """

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if self._title:
            title_lbl = QLabel(f"<b>{self._title}</b>")
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_lbl.setFont(QFont("Segoe UI", 11))
            title_lbl.setStyleSheet("color: #ccc; padding: 2px 0;")
            layout.addWidget(title_lbl)

        display_layout = QHBoxLayout()
        display_layout.setSpacing(6)

        self._original_label = QLabel("Original")
        self._original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._original_label.setMinimumSize(160, 100)
        self._original_label.setStyleSheet(
            "background-color: #111; border: 1px solid #2a2a2a; border-radius: 8px; color: #555; font-size: 11px;"
        )
        self._original_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        display_layout.addWidget(self._original_label)

        self._lut_label = QLabel("LUT")
        self._lut_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lut_label.setMinimumSize(160, 100)
        self._lut_label.setStyleSheet(
            "background-color: #111; border: 1px solid #7C5CFC; border-radius: 8px; color: #555; font-size: 11px;"
        )
        self._lut_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        display_layout.addWidget(self._lut_label)

        layout.addLayout(display_layout)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self._play_btn = QPushButton("\u25B6")
        self._play_btn.setObjectName("playBtn")
        self._play_btn.setFixedWidth(44)
        self._play_btn.setFixedHeight(32)
        self._play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self._play_btn)

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(1000)
        self._seek_slider.setValue(0)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        controls.addWidget(self._seek_slider)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setFont(QFont("Segoe UI", 9))
        self._time_label.setStyleSheet("color: #999;")
        self._time_label.setFixedWidth(110)
        controls.addWidget(self._time_label)

        self._frame_label = QLabel("F: 0")
        self._frame_label.setFont(QFont("Segoe UI", 9))
        self._frame_label.setStyleSheet("color: #777;")
        self._frame_label.setFixedWidth(60)
        controls.addWidget(self._frame_label)

        layout.addLayout(controls)

    def _open_video(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            self._original_label.setText("Cannot open video")
            self._cap = None
            return
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0
        self._target_delay_ms = max(16, int(1000.0 / self._fps))
        self._current_frame_idx = 0
        total_sec = self._total_frames / self._fps if self._fps > 0 else 0
        self._time_label.setText(f"00:00 / {self._format_time(total_sec)}")
        self._show_frame(0)

    def _read_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _resize_to_fit(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= self._max_width:
            return img
        aspect = h / w
        new_w = self._max_width
        new_h = int(new_w * aspect)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _array_to_pixmap(self, img: np.ndarray) -> QPixmap:
        h, w = img.shape[:2]
        qimg = QImage(img.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _show_frame(self, frame_idx: int) -> None:
        if self._closed or self._cap is None or not self._cap.isOpened():
            return
        try:
            frame_rgb = self._read_frame_at(frame_idx)
            if frame_rgb is None:
                return
            frame_rgb = self._resize_to_fit(frame_rgb)
            display_h = frame_rgb.shape[0]

            orig_pix = self._array_to_pixmap(frame_rgb)
            self._original_label.setPixmap(orig_pix)
            self._original_label.setFixedHeight(display_h)

            if self._use_lut and self._lut is not None:
                try:
                    from core.lut_generator import LUTGenerator
                    applied = LUTGenerator._apply_lut_trilinear(frame_rgb, self._lut)
                    lut_pix = self._array_to_pixmap(applied)
                    self._lut_label.setPixmap(lut_pix)
                except Exception:
                    self._lut_label.setPixmap(orig_pix)
            else:
                self._lut_label.setPixmap(orig_pix)
            self._lut_label.setFixedHeight(display_h)
        except Exception:
            pass

    def _next_frame(self) -> None:
        if self._closed or self._cap is None or not self._playing:
            return
        try:
            self._current_frame_idx += 1
            if self._current_frame_idx >= self._total_frames:
                self._current_frame_idx = 0
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._show_frame(self._current_frame_idx)
            self._update_controls()
        except Exception:
            pass

    def _update_controls(self) -> None:
        total_sec = self._total_frames / self._fps if self._fps > 0 else 0
        current_sec = self._current_frame_idx / self._fps if self._fps > 0 else 0
        self._time_label.setText(
            f"{self._format_time(current_sec)} / {self._format_time(total_sec)}"
        )
        self._frame_label.setText(f"F: {self._current_frame_idx}")
        progress = int((self._current_frame_idx / max(self._total_frames - 1, 1)) * 1000)
        self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(progress)
        self._seek_slider.blockSignals(False)

    def _on_seek(self, value: int) -> None:
        if self._closed or self._cap is None or not self._cap.isOpened():
            return
        try:
            target_frame = int((value / 1000.0) * (self._total_frames - 1))
            target_frame = max(0, min(target_frame, self._total_frames - 1))
            self._current_frame_idx = target_frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self._show_frame(target_frame)
            self._update_controls()
        except Exception:
            pass

    def toggle_play(self) -> None:
        if self._closed:
            return
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self._closed or self._cap is None or self._total_frames == 0:
            return
        if self._current_frame_idx >= self._total_frames - 1:
            self._current_frame_idx = 0
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
        self._playing = True
        self._play_btn.setText("\u23F8")
        self._timer.start(self._target_delay_ms)

    def pause(self) -> None:
        self._playing = False
        self._play_btn.setText("\u25B6")
        self._timer.stop()

    def set_lut(self, lut: np.ndarray) -> None:
        self._lut = lut
        if not self._closed:
            self._show_frame(self._current_frame_idx)

    def set_use_lut(self, enabled: bool) -> None:
        self._use_lut = enabled
        if not self._closed:
            self._show_frame(self._current_frame_idx)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._playing = False
        try:
            self._timer.timeout.disconnect(self._next_frame)
        except Exception:
            pass
        self._timer.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    @staticmethod
    def _format_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


class LUTComparisonCard(QWidget):
    """A card showing a single LUT video player with its parameters and save button."""

    save_requested = pyqtSignal(np.ndarray, dict)

    def __init__(
        self,
        video_path: str,
        lut_data: np.ndarray,
        params: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._lut_data = lut_data
        self._params = params
        self.setObjectName("playerCard")
        self.setStyleSheet(
            "#playerCard { background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        size = params.get("size", 33)
        strength = params.get("strength", 1.0)
        mode = params.get("mode", "natural")
        mode_icon = "\u2696" if mode == "natural" else "\u26A1"

        header = QHBoxLayout()
        info_lbl = QLabel(f"<b>{mode_icon} {size}x{size}x{size}</b>  \u00B7  {mode.title()}  \u00B7  {strength:.0%}")
        info_lbl.setFont(QFont("Segoe UI", 10))
        info_lbl.setStyleSheet("color: #bbb;")
        header.addWidget(info_lbl)
        header.addStretch()

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("saveBtn")
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self._save_btn)

        layout.addLayout(header)

        self.player = LUTVideoPlayer(
            video_path=video_path,
            lut=lut_data,
            title="",
            max_width=360,
            parent=self,
        )
        layout.addWidget(self.player)

    def _on_save(self) -> None:
        self.save_requested.emit(self._lut_data, self._params)


class ComparisonDialog(QDialog):
    """Comparison view showing multiple LUTs applied to the same video."""

    def __init__(
        self,
        video_path: str,
        luts: List[dict],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("LUT Comparison")
        self.setMinimumSize(1100, 650)
        self.setStyleSheet(DARK_STYLE)

        self._video_path = video_path
        self._luts = luts
        self._cards: List[LUTComparisonCard] = []

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("<h1 style='font-weight: 300; letter-spacing: 1px;'>Compare LUTs</h1>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 18))
        title.setStyleSheet("color: #f5f5f5;")
        layout.addWidget(title)

        subtitle = QLabel("Compare different LUT settings side by side. Click <b>Save</b> on any LUT to export it as .cube.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888; font-size: 12px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        n = len(self._luts)
        cols = max(1, min(2, n))

        row_layout = None
        for i, lut_entry in enumerate(self._luts):
            if i % cols == 0:
                row_layout = QHBoxLayout()
                row_layout.setSpacing(20)
                scroll_layout.addLayout(row_layout)

            card = LUTComparisonCard(
                video_path=self._video_path,
                lut_data=lut_entry["lut"],
                params=lut_entry["params"],
            )
            card.save_requested.connect(self._on_save_lut)
            self._cards.append(card)
            row_layout.addWidget(card)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._play_all_btn = QPushButton("Play All")
        self._play_all_btn.setObjectName("successBtn")
        self._play_all_btn.clicked.connect(self._play_all)
        self._play_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_layout.addWidget(self._play_all_btn)

        self._pause_all_btn = QPushButton("Pause All")
        self._pause_all_btn.clicked.connect(self._pause_all)
        self._pause_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_layout.addWidget(self._pause_all_btn)

        btn_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setObjectName("dangerBtn")
        close_btn.clicked.connect(self.close)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _play_all(self) -> None:
        for card in self._cards:
            card.player.play()

    def _pause_all(self) -> None:
        for card in self._cards:
            card.player.pause()

    def _on_save_lut(self, lut_data: np.ndarray, params: dict) -> None:
        size = params.get("size", 33)
        strength = params.get("strength", 1.0)
        mode = params.get("mode", "natural")
        suggested = f"lut_{mode}_{size}x{size}x{size}_{strength:.0%}.cube"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LUT as .cube",
            suggested,
            "Cube LUT (*.cube);;All Files (*)",
        )
        if not path:
            return

        try:
            CubeExporter.export(lut_data, path, title="Video LUT Generator")
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

    def closeEvent(self, event) -> None:
        for card in self._cards:
            card.player.shutdown()
        super().closeEvent(event)
