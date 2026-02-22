from __future__ import annotations
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Any
import copy
import shutil
import tempfile
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from lipidbench.gui.pandas_table_model import PandasTableModel
from lipidbench.utils.feature_table_io import (
    build_feature_view_table,
    find_feature_table,
    load_feature_table,
    normalize_results_base_dir,
    standardize_rt_columns_for_display,
)
from lipidbench.utils.feature_alignment import align_feature_tables, load_and_standardize_table, missing_features_for_algo
from lipidbench.utils.peak_attributes import compute_peak_attributes


def _check_pyopenms_available(python_executable: Optional[str] = None) -> tuple[bool, str]:
    exe = str(python_executable).strip() if python_executable else sys.executable
    check_code = (
        "import os,site,sysconfig;"
        "c=[];"
        "[c.extend([__import__('pathlib').Path(sp)/'pyopenms', __import__('pathlib').Path(sp)/'pyopenms.libs']) for sp in site.getsitepackages()];"
        "pl=sysconfig.get_paths().get('platlib');"
        "c.extend([__import__('pathlib').Path(pl)/'pyopenms', __import__('pathlib').Path(pl)/'pyopenms.libs']) if pl else None;"
        "[(hasattr(os,'add_dll_directory') and os.add_dll_directory(str(d))) for d in c if getattr(d,'exists',lambda:False)()];"
        "import pyopenms;print(pyopenms.__version__)"
    )
    try:
        proc = subprocess.run(
            [
                exe,
                "-c",
                check_code,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        _ = proc.stdout.strip()
        return True, ""
    except Exception as e:
        detail = (
            f"pyopenms 检查失败（子进程）。\n"
            f"解释器: {exe}\n"
            f"原始错误: {e}"
        )
        return False, detail


class _SimpleSpectrum:
    """Lightweight spectrum adapter for EIC extraction fallback."""

    def __init__(self, rt_min: float, mz: np.ndarray, intensity: np.ndarray, ms_level: int = 1):
        self._rt_sec = float(rt_min) * 60.0
        self._mz = np.asarray(mz, dtype=np.float64)
        self._intensity = np.asarray(intensity, dtype=np.float64)
        self._ms_level = int(ms_level)

    def getMSLevel(self) -> int:
        return self._ms_level

    def getRT(self) -> float:
        return self._rt_sec

    def get_peaks(self):
        return self._mz, self._intensity


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LipidBench EIC Viewer")
        self.resize(1200, 700)

        try:
            from lipidbench.utils.config_io import load_config

            self._config_defaults: dict[str, Any] = load_config()
        except Exception:
            self._config_defaults = {}

        self._feature_df_raw: Optional[pd.DataFrame] = None
        self._feature_df_view: pd.DataFrame = pd.DataFrame(columns=["Feature_ID", "mz", "RT", "RTmin", "RTmax", "PeakArea"])
        self._mzml_path: Optional[Path] = None
        self._ms_exp = None
        self._mzml_ready = False
        self._last_feature_id: Optional[str] = None
        self._trace_job_id = 0
        self._trace_busy = False
        self._trace_pending: Optional[tuple[float, float, Optional[float], Optional[float], str]] = None
        self._trace_thread: Optional[QtCore.QThread] = None
        self._trace_worker: Optional[QtCore.QObject] = None
        self._batch_thread: Optional[QtCore.QThread] = None
        self._batch_worker: Optional[QtCore.QObject] = None
        self._build_ui()

    def _get_pyopenms_python(self) -> Optional[str]:
        runtime_cfg = self._config_defaults.get("runtime", {})
        if not isinstance(runtime_cfg, dict):
            return None
        candidate = str(runtime_cfg.get("pyopenms_python", "")).strip()
        if not candidate:
            return None
        p = Path(candidate)
        if not p.exists():
            return None
        return str(p)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        header_box = QtWidgets.QGroupBox("运行")
        header_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        header_layout = QtWidgets.QVBoxLayout(header_box)
        header_layout.setContentsMargins(10, 8, 10, 8)

        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(["asari", "pyopenms", "xcms", "msdial"])
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)

        self.input_mode_combo = QtWidgets.QComboBox()
        self.input_mode_combo.addItem("单文件", "single")
        self.input_mode_combo.addItem("多文件文件夹", "folder")
        self.input_mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)

        self.mzml_path_edit = QtWidgets.QLineEdit()
        self.mzml_path_edit.setPlaceholderText("选择单个 mzML 文件")
        self.mzml_browse_btn = QtWidgets.QPushButton("选择…")
        self.mzml_browse_btn.clicked.connect(self._browse_mzml)
        mzml_row = QtWidgets.QHBoxLayout()
        mzml_row.addWidget(self.mzml_path_edit, 1)
        mzml_row.addWidget(self.mzml_browse_btn)
        self.mzml_row_w = QtWidgets.QWidget()
        self.mzml_row_w.setLayout(mzml_row)

        self.mzml_dir_edit = QtWidgets.QLineEdit()
        self.mzml_dir_edit.setPlaceholderText("选择包含多个 mzML 的目录")
        self.mzml_dir_browse_btn = QtWidgets.QPushButton("选择…")
        self.mzml_dir_browse_btn.clicked.connect(self._browse_mzml_dir)
        mzml_dir_row = QtWidgets.QHBoxLayout()
        mzml_dir_row.addWidget(self.mzml_dir_edit, 1)
        mzml_dir_row.addWidget(self.mzml_dir_browse_btn)
        self.mzml_dir_row_w = QtWidgets.QWidget()
        self.mzml_dir_row_w.setLayout(mzml_dir_row)

        self.results_dir_edit = QtWidgets.QLineEdit()
        self.results_dir_edit.setPlaceholderText("选择导出结果目录")
        self.results_browse_btn = QtWidgets.QPushButton("选择…")
        self.results_browse_btn.clicked.connect(self._browse_results_dir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.results_dir_edit, 1)
        out_row.addWidget(self.results_browse_btn)
        out_row_w = QtWidgets.QWidget()
        out_row_w.setLayout(out_row)

        self.msdial_xlsx_edit = QtWidgets.QLineEdit()
        self.msdial_xlsx_edit.setPlaceholderText("选择 MS-DIAL 导出的 xlsx/xls")
        self.msdial_xlsx_browse_btn = QtWidgets.QPushButton("选择…")
        self.msdial_xlsx_browse_btn.clicked.connect(self._browse_msdial_xlsx)
        msdial_row = QtWidgets.QHBoxLayout()
        msdial_row.addWidget(self.msdial_xlsx_edit, 1)
        msdial_row.addWidget(self.msdial_xlsx_browse_btn)
        self.msdial_row_w = QtWidgets.QWidget()
        self.msdial_row_w.setLayout(msdial_row)

        self.asari_table_combo = QtWidgets.QComboBox()
        self.asari_table_combo.addItems(["preferred", "full"])

        # xcms params
        xcms_peak = self._config_defaults.get("parameters", {}).get("xcms", {}).get("peak_picking", {})
        xcms_peakwidth = xcms_peak.get("peakwidth", [5, 50])
        self.xcms_ppm_spin = QtWidgets.QDoubleSpinBox()
        self.xcms_ppm_spin.setRange(0.1, 500.0)
        self.xcms_ppm_spin.setValue(float(xcms_peak.get("ppm", 10)))
        self.xcms_noise_spin = QtWidgets.QDoubleSpinBox()
        self.xcms_noise_spin.setRange(0.0, 1e12)
        self.xcms_noise_spin.setDecimals(1)
        self.xcms_noise_spin.setValue(float(xcms_peak.get("noise", 1000)))
        self.xcms_sn_spin = QtWidgets.QDoubleSpinBox()
        self.xcms_sn_spin.setRange(0.0, 200.0)
        self.xcms_sn_spin.setValue(float(xcms_peak.get("snthresh", 3)))
        self.xcms_minwidth_spin = QtWidgets.QDoubleSpinBox()
        self.xcms_minwidth_spin.setRange(0.1, 5000.0)
        self.xcms_minwidth_spin.setValue(float(xcms_peakwidth[0] if len(xcms_peakwidth) == 2 else 5))
        self.xcms_maxwidth_spin = QtWidgets.QDoubleSpinBox()
        self.xcms_maxwidth_spin.setRange(0.1, 5000.0)
        self.xcms_maxwidth_spin.setValue(float(xcms_peakwidth[1] if len(xcms_peakwidth) == 2 else 50))

        # pyopenms params
        pyopenms_cfg = self._config_defaults.get("parameters", {}).get("pyopenms", {})
        pyopenms_peak = pyopenms_cfg.get("peak_picking", {})
        self.pyopenms_ppm_spin = QtWidgets.QDoubleSpinBox()
        self.pyopenms_ppm_spin.setRange(0.1, 500.0)
        self.pyopenms_ppm_spin.setValue(float(pyopenms_peak.get("mz_tol", pyopenms_cfg.get("mz_tol", 10.0))))
        self.pyopenms_noise_spin = QtWidgets.QDoubleSpinBox()
        self.pyopenms_noise_spin.setRange(0.0, 1e12)
        self.pyopenms_noise_spin.setDecimals(1)
        self.pyopenms_noise_spin.setValue(float(pyopenms_peak.get("noise", pyopenms_cfg.get("noise", 1000))))
        self.pyopenms_sn_spin = QtWidgets.QDoubleSpinBox()
        self.pyopenms_sn_spin.setRange(0.0, 200.0)
        self.pyopenms_sn_spin.setValue(float(pyopenms_peak.get("sn", pyopenms_cfg.get("sn", 5))))
        self.pyopenms_min_fwhm_spin = QtWidgets.QDoubleSpinBox()
        self.pyopenms_min_fwhm_spin.setRange(0.1, 5000.0)
        self.pyopenms_min_fwhm_spin.setValue(float(pyopenms_peak.get("min_fwhm", pyopenms_cfg.get("min_fwhm", 2.5))))
        self.pyopenms_max_fwhm_spin = QtWidgets.QDoubleSpinBox()
        self.pyopenms_max_fwhm_spin.setRange(0.1, 5000.0)
        self.pyopenms_max_fwhm_spin.setValue(float(pyopenms_peak.get("max_fwhm", pyopenms_cfg.get("max_fwhm", 60.0))))

        # asari params
        asari_cfg = self._config_defaults.get("parameters", {}).get("asari", {})
        self.asari_ppm_spin = QtWidgets.QDoubleSpinBox()
        self.asari_ppm_spin.setRange(0.1, 500.0)
        self.asari_ppm_spin.setValue(float(asari_cfg.get("ppm", 10.0)))
        self.asari_min_int_spin = QtWidgets.QDoubleSpinBox()
        self.asari_min_int_spin.setRange(0.0, 1e12)
        self.asari_min_int_spin.setDecimals(1)
        self.asari_min_int_spin.setValue(float(asari_cfg.get("min_intensity_threshold", 1000)))
        self.asari_min_height_spin = QtWidgets.QDoubleSpinBox()
        self.asari_min_height_spin.setRange(0.0, 1e12)
        self.asari_min_height_spin.setDecimals(1)
        self.asari_min_height_spin.setValue(float(asari_cfg.get("min_peak_height", 10000)))
        self.asari_mode_combo = QtWidgets.QComboBox()
        self.asari_mode_combo.addItems(["pos", "neg"])
        mode = str(asari_cfg.get("mode", "pos")).lower()
        self.asari_mode_combo.setCurrentText("neg" if mode == "neg" else "pos")
        self.asari_autoheight_check = QtWidgets.QCheckBox("autoheight")
        self.asari_autoheight_check.setChecked(bool(asari_cfg.get("autoheight", True)))

        self.ppm_spin = QtWidgets.QDoubleSpinBox()
        self.ppm_spin.setRange(0.1, 200.0)
        self.ppm_spin.setDecimals(2)
        self.ppm_spin.setValue(10.0)

        self.run_btn = QtWidgets.QPushButton("运行并加载")
        self.run_btn.clicked.connect(self._run_and_load)

        self.export_plot_btn = QtWidgets.QPushButton("导出当前图像")
        self.export_plot_btn.setEnabled(False)
        self.export_plot_btn.clicked.connect(self._export_current_plot)

        self.export_batch_btn = QtWidgets.QPushButton("批量导出 EIC 图")
        self.export_batch_btn.clicked.connect(lambda: self._export_batch_eic(auto=False))

        self.auto_export_check = QtWidgets.QCheckBox("特征检测完成后自动导出EIC图")
        self.auto_export_check.setChecked(False)

        self.export_scope_combo = QtWidgets.QComboBox()
        self.export_scope_combo.addItem("全部", "all")
        self.export_scope_combo.addItem("前N个", "topn")

        self.export_topn_spin = QtWidgets.QSpinBox()
        self.export_topn_spin.setRange(1, 1000000)
        self.export_topn_spin.setValue(200)
        self.export_scope_combo.currentIndexChanged.connect(
            lambda _: self.export_topn_spin.setEnabled(self.export_scope_combo.currentData() == "topn")
        )
        self.export_topn_spin.setEnabled(False)

        self.export_all_attr_btn = QtWidgets.QPushButton("导出峰属性（当前算法全部特征）")
        self.export_all_attr_btn.clicked.connect(self._export_all_peak_attributes)
        self.export_missing_attr_btn = QtWidgets.QPushButton("导出峰属性（多算法对齐后未检出）")
        self.export_missing_attr_btn.clicked.connect(self._export_missing_peak_attributes)

        self.params_toggle_btn = QtWidgets.QToolButton()
        self.params_toggle_btn.setText("显示当前算法参数")
        self.params_toggle_btn.setCheckable(True)

        self.params_container = QtWidgets.QWidget()
        params_layout = QtWidgets.QVBoxLayout(self.params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_stack = QtWidgets.QStackedWidget()
        params_layout.addWidget(self.params_stack)
        self.params_container.setVisible(False)
        self.params_toggle_btn.toggled.connect(self._on_toggle_params)

        # Per-algorithm param pages (only one page visible at a time)
        xcms_page = QtWidgets.QWidget()
        xcms_form = QtWidgets.QFormLayout(xcms_page)
        xcms_form.setContentsMargins(0, 0, 0, 0)
        xcms_form.addRow("XCMS ppm", self.xcms_ppm_spin)
        xcms_form.addRow("XCMS noise", self.xcms_noise_spin)
        xcms_form.addRow("XCMS SN", self.xcms_sn_spin)
        xcms_form.addRow("XCMS min peakwidth", self.xcms_minwidth_spin)
        xcms_form.addRow("XCMS max peakwidth", self.xcms_maxwidth_spin)

        pyopenms_page = QtWidgets.QWidget()
        pyopenms_form = QtWidgets.QFormLayout(pyopenms_page)
        pyopenms_form.setContentsMargins(0, 0, 0, 0)
        pyopenms_form.addRow("pyopenms ppm", self.pyopenms_ppm_spin)
        pyopenms_form.addRow("pyopenms noise", self.pyopenms_noise_spin)
        pyopenms_form.addRow("pyopenms SN", self.pyopenms_sn_spin)
        pyopenms_form.addRow("pyopenms min FWHM", self.pyopenms_min_fwhm_spin)
        pyopenms_form.addRow("pyopenms max FWHM", self.pyopenms_max_fwhm_spin)

        asari_page = QtWidgets.QWidget()
        asari_form = QtWidgets.QFormLayout(asari_page)
        asari_form.setContentsMargins(0, 0, 0, 0)
        asari_form.addRow("Asari 表", self.asari_table_combo)
        asari_form.addRow("asari ppm", self.asari_ppm_spin)
        asari_form.addRow("asari min intensity", self.asari_min_int_spin)
        asari_form.addRow("asari min peak height", self.asari_min_height_spin)
        asari_form.addRow("asari mode", self.asari_mode_combo)
        asari_form.addRow("asari autoheight", self.asari_autoheight_check)

        msdial_page = QtWidgets.QWidget()
        msdial_form = QtWidgets.QFormLayout(msdial_page)
        msdial_form.setContentsMargins(0, 0, 0, 0)
        msdial_form.addRow(QtWidgets.QLabel("MS-DIAL 无额外算法参数"))

        self.params_stack.addWidget(asari_page)    # 0
        self.params_stack.addWidget(pyopenms_page) # 1
        self.params_stack.addWidget(xcms_page)     # 2
        self.params_stack.addWidget(msdial_page)   # 3

        tabs = QtWidgets.QTabWidget()

        basic_tab = QtWidgets.QWidget()
        basic_form = QtWidgets.QFormLayout(basic_tab)
        basic_form.setContentsMargins(6, 6, 6, 6)
        basic_form.addRow("算法", self.algo_combo)
        basic_form.addRow("输入模式", self.input_mode_combo)
        basic_form.addRow("mzML", self.mzml_row_w)
        basic_form.addRow("mzML目录", self.mzml_dir_row_w)
        basic_form.addRow("输出目录", out_row_w)
        basic_form.addRow("MS-DIAL 表", self.msdial_row_w)
        basic_form.addRow("EIC ppm", self.ppm_spin)

        params_tab = QtWidgets.QWidget()
        params_layout_tab = QtWidgets.QVBoxLayout(params_tab)
        params_layout_tab.setContentsMargins(6, 6, 6, 6)
        params_layout_tab.addWidget(self.params_toggle_btn, 0)
        params_layout_tab.addWidget(self.params_container, 0)
        params_layout_tab.addStretch(1)

        export_tab = QtWidgets.QWidget()
        export_form = QtWidgets.QFormLayout(export_tab)
        export_form.setContentsMargins(6, 6, 6, 6)
        export_form.addRow("导出时机", self.auto_export_check)
        export_form.addRow("导出范围", self.export_scope_combo)
        export_form.addRow("导出N", self.export_topn_spin)
        export_form.addRow("", self.export_batch_btn)
        export_form.addRow("", self.export_all_attr_btn)
        export_form.addRow("", self.export_missing_attr_btn)

        tabs.addTab(basic_tab, "基础")
        tabs.addTab(params_tab, "算法参数")
        tabs.addTab(export_tab, "导出")

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.export_plot_btn)

        header_layout.addWidget(tabs)
        header_layout.addLayout(btn_row)
        self._on_input_mode_changed()
        self._on_algo_changed(self.algo_combo.currentText())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table = QtWidgets.QTableView()
        self.model = PandasTableModel(self._feature_df_view)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableView::item:selected {"
            " background-color: #2E7DFF;"
            " color: white;"
            "}"
            "QTableView::item:selected:active {"
            " background-color: #1E5FE0;"
            " color: white;"
            "}"
        )
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)

        table_wrap = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_wrap)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QtWidgets.QLabel("特征列表"))
        table_layout.addWidget(self.table)

        plot_wrap = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_wrap)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
        except Exception as e:
            msg = QtWidgets.QLabel(f"matplotlib Qt backend 不可用：{e}")
            msg.setWordWrap(True)
            plot_layout.addWidget(msg)
            self.canvas = None
            self.toolbar = None
            self.fig = None
            self.ax = None
        else:
            self.fig = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.canvas.customContextMenuRequested.connect(self._on_canvas_context_menu)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.ax = self.fig.add_subplot(111)
            plot_layout.addWidget(QtWidgets.QLabel("EIC 图"))
            plot_layout.addWidget(self.toolbar)
            plot_layout.addWidget(self.canvas, 1)

        splitter.addWidget(table_wrap)
        splitter.addWidget(plot_wrap)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(header_box)
        layout.addWidget(splitter, 1)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def _on_toggle_params(self, checked: bool) -> None:
        self.params_container.setVisible(checked)
        self.params_toggle_btn.setText("隐藏当前算法参数" if checked else "显示当前算法参数")

    def _browse_results_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择结果目录", str(Path.cwd()))
        if path:
            self.results_dir_edit.setText(path)

    def _browse_mzml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 mzML", str(Path.cwd()), "mzML (*.mzML)")
        if path:
            self.mzml_path_edit.setText(path)

    def _browse_mzml_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择包含 mzML 的目录", str(Path.cwd()))
        if path:
            self.mzml_dir_edit.setText(path)

    def _browse_msdial_xlsx(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 MS-DIAL 导出的 xlsx", str(Path.cwd()), "Excel (*.xlsx *.xls)")
        if path:
            self.msdial_xlsx_edit.setText(path)

    def _on_algo_changed(self, algo_text: str) -> None:
        algo = algo_text.strip().lower()

        is_msdial = algo == "msdial"
        self.msdial_row_w.setVisible(is_msdial)

        if algo == "asari":
            self.params_stack.setCurrentIndex(0)
        elif algo == "pyopenms":
            self.params_stack.setCurrentIndex(1)
        elif algo == "xcms":
            self.params_stack.setCurrentIndex(2)
        else:
            self.params_stack.setCurrentIndex(3)

    def _on_input_mode_changed(self) -> None:
        mode = self.input_mode_combo.currentData()
        is_folder = mode == "folder"
        self.mzml_row_w.setVisible(not is_folder)
        self.mzml_dir_row_w.setVisible(is_folder)

    def _collect_algo_params(self, algo: str) -> dict[str, float | str | bool]:
        if algo == "xcms":
            return {
                "mz_tol": float(self.xcms_ppm_spin.value()),
                "noise": float(self.xcms_noise_spin.value()),
                "sn": float(self.xcms_sn_spin.value()),
                "minwidth": float(self.xcms_minwidth_spin.value()),
                "maxwidth": float(self.xcms_maxwidth_spin.value()),
            }
        if algo == "pyopenms":
            return {
                "mz_tol": float(self.pyopenms_ppm_spin.value()),
                "noise": float(self.pyopenms_noise_spin.value()),
                "sn": float(self.pyopenms_sn_spin.value()),
                "min_fwhm": float(self.pyopenms_min_fwhm_spin.value()),
                "max_fwhm": float(self.pyopenms_max_fwhm_spin.value()),
            }
        if algo == "asari":
            return {
                "ppm": float(self.asari_ppm_spin.value()),
                "min_intensity_threshold": float(self.asari_min_int_spin.value()),
                "mode": self.asari_mode_combo.currentText().strip().lower(),
                "autoheight": bool(self.asari_autoheight_check.isChecked()),
            }
        return {}

    def _run_and_load(self) -> None:
        algo = self.algo_combo.currentText().strip().lower()
        input_mode = self.input_mode_combo.currentData()
        mzml_path = Path(self.mzml_path_edit.text().strip()) if self.mzml_path_edit.text().strip() else None
        mzml_dir = Path(self.mzml_dir_edit.text().strip()) if self.mzml_dir_edit.text().strip() else None
        results_dir = Path(self.results_dir_edit.text().strip()) if self.results_dir_edit.text().strip() else None
        algo_params = self._collect_algo_params(algo)
        asari_table_preference = self.asari_table_combo.currentText().strip().lower()
        input_files: list[Path] = []

        if algo != "msdial":
            if input_mode == "folder":
                if mzml_dir is None or not mzml_dir.exists() or not mzml_dir.is_dir():
                    self.status.showMessage("mzML 目录无效", 5000)
                    return
                input_files = sorted(mzml_dir.glob("*.mzML"))
                if not input_files:
                    self.status.showMessage("mzML 目录中没有 .mzML 文件", 6000)
                    return
            else:
                if mzml_path is None or not mzml_path.exists():
                    self.status.showMessage("mzML 路径无效", 5000)
                    return
                input_files = [mzml_path]
        else:
            if mzml_path is None or not mzml_path.exists():
                self.status.showMessage("未提供 mzML，将只显示特征表，无法绘制 EIC", 7000)
        if results_dir is None:
            self.status.showMessage("输出目录无效", 5000)
            return

        results_dir = normalize_results_base_dir(results_dir, algo)
        results_dir.mkdir(parents=True, exist_ok=True)

        if algo == "msdial":
            xlsx_text = self.msdial_xlsx_edit.text().strip()
            if not xlsx_text:
                xlsx, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 MS-DIAL 导出的 xlsx", str(results_dir), "Excel (*.xlsx *.xls)")
                if xlsx:
                    self.msdial_xlsx_edit.setText(xlsx)
                    xlsx_text = xlsx
            if not xlsx_text:
                return
            self._start_worker(
                algo=algo,
                input_files=input_files,
                input_mode=input_mode,
                mzml_dir=mzml_dir,
                results_dir=results_dir,
                msdial_xlsx=Path(xlsx_text),
                algo_params=algo_params,
                asari_table_preference=asari_table_preference,
                pyopenms_python=None,
            )
            return

        if algo == "pyopenms":
            pyopenms_python = self._get_pyopenms_python()
            ok, detail = _check_pyopenms_available(pyopenms_python)
            if not ok:
                QtWidgets.QMessageBox.critical(self, "pyopenms 无法导入", detail)
                self.status.showMessage("pyopenms 导入失败（已弹窗显示详情）", 10000)
                return
            if pyopenms_python:
                self.status.showMessage(f"pyopenms 将使用外部解释器: {pyopenms_python}", 6000)

        self._start_worker(
            algo=algo,
            input_files=input_files,
            input_mode=input_mode,
            mzml_dir=mzml_dir,
            results_dir=results_dir,
            msdial_xlsx=None,
            algo_params=algo_params,
            asari_table_preference=asari_table_preference,
            pyopenms_python=self._get_pyopenms_python(),
        )

    def _start_worker(
        self,
        *,
        algo: str,
        input_files: list[Path],
        input_mode: str,
        mzml_dir: Optional[Path],
        results_dir: Path,
        msdial_xlsx: Optional[Path],
        algo_params: dict[str, float | str | bool],
        asari_table_preference: str,
        pyopenms_python: Optional[str],
    ) -> None:
        self.run_btn.setEnabled(False)
        self.status.showMessage("运行算法中…")

        class Worker(QtCore.QObject):
            finished = QtCore.Signal(str, object)
            failed = QtCore.Signal(str)

            def __init__(
                self,
                algo: str,
                input_files: list[Path],
                input_mode: str,
                mzml_dir: Optional[Path],
                results_dir: Path,
                msdial_xlsx: Optional[Path],
                algo_params: dict[str, float | str | bool],
                asari_table_preference: str,
                pyopenms_python: Optional[str],
            ):
                super().__init__()
                self.algo = algo
                self.input_files = input_files
                self.input_mode = input_mode
                self.mzml_dir = mzml_dir
                self.results_dir = results_dir
                self.msdial_xlsx = msdial_xlsx
                self.algo_params = algo_params
                self.asari_table_preference = asari_table_preference
                self.pyopenms_python = pyopenms_python

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    df = self._run_impl()
                except Exception as e:
                    self.failed.emit(str(e))
                    return
                self.finished.emit(self.algo, df)

            def _run_impl(self) -> pd.DataFrame:
                from lipidbench.utils.config_io import load_config
                from lipidbench.utils.data_io import load_msdial_results, load_xcms_results, load_pyopenms_results
                from lipidbench.runners.run_xcms import extract_xcms_params, run_xcms
                from lipidbench.runners.run_pyopenms import extract_pyopenms_params, run_pyopenms
                from lipidbench.runners.run_asari import run_asari_pipeline

                config = load_config()
                algo = self.algo
                results_dir = normalize_results_base_dir(Path(self.results_dir), algo).resolve()
                results_dir.mkdir(parents=True, exist_ok=True)

                if algo == "msdial":
                    if self.msdial_xlsx is None or not self.msdial_xlsx.exists():
                        raise FileNotFoundError("MS-DIAL xlsx not provided")
                    out_dir = results_dir / "msdial"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{self.msdial_xlsx.stem}_processed.csv"
                    load_msdial_results(self.msdial_xlsx, out_file)
                    return load_feature_table(out_file, algo)

                if not self.input_files:
                    raise FileNotFoundError("未找到可用 mzML 输入")

                if algo == "xcms":
                    with tempfile.TemporaryDirectory(prefix="tmp_mzml_") as tmp_dir_str:
                        tmp_dir = Path(tmp_dir_str)
                        for p in self.input_files:
                            shutil.copy2(p, tmp_dir / p.name)
                        out_file = results_dir / "xcms" / "xcms_features.csv"
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        params = extract_xcms_params(config)
                        params.update({k: v for k, v in self.algo_params.items() if k in {"mz_tol", "noise", "sn", "minwidth", "maxwidth"}})
                        run_xcms(input_dir=tmp_dir, output_file=out_file, **params)
                        load_xcms_results(out_file)
                        return load_feature_table(out_file, algo)

                if algo == "asari":
                    with tempfile.TemporaryDirectory(prefix="tmp_mzml_") as tmp_dir_str:
                        tmp_dir = Path(tmp_dir_str)
                        for p in self.input_files:
                            shutil.copy2(p, tmp_dir / p.name)
                        cfg = copy.deepcopy(config)
                        cfg.setdefault("paths", {})
                        cfg.setdefault("parameters", {})
                        cfg["parameters"].setdefault("asari", {})
                        for k in ["ppm", "min_intensity_threshold", "mode", "autoheight"]:
                            if k in self.algo_params:
                                cfg["parameters"]["asari"][k] = self.algo_params[k]
                        cfg["paths"]["input_dir"] = str(tmp_dir)
                        cfg["paths"]["asari_output"] = str((results_dir / "asari").resolve())
                        run_asari_pipeline(cfg)
                        asari_dir = results_dir / "asari"
                        preferred = asari_dir / "preferred_Feature_table.csv"
                        full = asari_dir / "full_Feature_table.csv"
                        if self.asari_table_preference == "full" and full.exists():
                            feature_path = full
                        elif self.asari_table_preference == "preferred" and preferred.exists():
                            feature_path = preferred
                        else:
                            feature_path = find_feature_table(results_dir, algo)
                        return load_feature_table(feature_path, algo)

                if algo == "pyopenms":
                    with tempfile.TemporaryDirectory(prefix="tmp_mzml_") as tmp_dir_str:
                        tmp_dir = Path(tmp_dir_str)
                        for p in self.input_files:
                            shutil.copy2(p, tmp_dir / p.name)

                        out_file = results_dir / "pyopenms" / "pyopenms_features.csv"
                        out_file.parent.mkdir(parents=True, exist_ok=True)

                        params = extract_pyopenms_params(config)
                        params.update(
                            {
                                k: v
                                for k, v in self.algo_params.items()
                                if k in {"mz_tol", "noise", "sn", "min_fwhm", "max_fwhm"}
                            }
                        )

                        if self.pyopenms_python:
                            from lipidbench.runners.run_pyopenms import run_pyopenms_subprocess

                            run_pyopenms_subprocess(
                                input_dir=tmp_dir,
                                output_file=out_file,
                                python_executable=self.pyopenms_python,
                                **params,
                            )
                        else:
                            run_pyopenms(input_dir=tmp_dir, output_file=out_file, **params)
                        load_pyopenms_results(out_file, input_dir=tmp_dir, **params)
                        return load_feature_table(out_file, algo)

                raise ValueError(f"Unknown algorithm: {algo}")

        self._worker_thread = QtCore.QThread(self)
        self._worker = Worker(algo, input_files, input_mode, mzml_dir, results_dir, msdial_xlsx, algo_params, asari_table_preference, pyopenms_python)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_worker_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "运行失败", msg)
        self.status.showMessage(f"运行失败：{msg}", 12000)

    def _on_worker_finished(self, algo: str, df: pd.DataFrame) -> None:
        self.run_btn.setEnabled(True)
        self._feature_df_raw = df
        self._last_feature_id = None
        self.export_plot_btn.setEnabled(False)
        self._recompute_view()

        if self.input_mode_combo.currentData() == "single":
            mzml_path = Path(self.mzml_path_edit.text().strip())
        else:
            mzml_dir = Path(self.mzml_dir_edit.text().strip()) if self.mzml_dir_edit.text().strip() else None
            mzml_path = None
            if mzml_dir is not None and mzml_dir.exists():
                files = sorted(mzml_dir.glob("*.mzML"))
                if files:
                    mzml_path = files[0]

        if mzml_path is not None and mzml_path.exists():
            self._mzml_path = mzml_path
            self._load_mzml()

        # 自动选中第一行并实时绘制，避免“有表无图”的感知问题。
        if not self._feature_df_view.empty:
            self.table.selectRow(0)
            QtCore.QTimer.singleShot(0, self._try_plot_selected_row)

        self.status.showMessage("运行完成并加载", 5000)

        if self.auto_export_check.isChecked():
            self._export_batch_eic(auto=True)

    def _recompute_view(self) -> None:
        if self._feature_df_raw is None:
            return
        algo = self.algo_combo.currentText().strip().lower()
        self._feature_df_view = build_feature_view_table(self._feature_df_raw, algo)

        self.model.set_dataframe(self._feature_df_view)
        self.table.resizeColumnsToContents()

    def _load_mzml(self) -> None:
        if self._mzml_path is None:
            return

        # GUI 预览固定使用 pymzml，彻底规避 pyopenms DLL 依赖问题。
        try:
            import pymzml  # type: ignore
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "EIC 预览不可用",
                f"pymzml 不可用：\n{e}",
            )
            self._ms_exp = None
            self._mzml_ready = False
            return

        try:
            reader = pymzml.run.Reader(str(self._mzml_path))
            specs: list[_SimpleSpectrum] = []
            for spec in reader:
                ms_level = int(getattr(spec, "ms_level", 1) or 1)
                if ms_level != 1:
                    continue

                try:
                    rt_min = float(spec.scan_time_in_minutes())
                except Exception:
                    continue

                try:
                    peaks = spec.peaks("centroided")
                except Exception:
                    peaks = spec.peaks("raw")

                if peaks is None or len(peaks) == 0:
                    mz = np.asarray([], dtype=np.float64)
                    intensity = np.asarray([], dtype=np.float64)
                else:
                    arr = np.asarray(peaks, dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        mz = np.asarray([], dtype=np.float64)
                        intensity = np.asarray([], dtype=np.float64)
                    else:
                        mz = arr[:, 0]
                        intensity = arr[:, 1]

                specs.append(_SimpleSpectrum(rt_min=rt_min, mz=mz, intensity=intensity, ms_level=1))

            if not specs:
                raise RuntimeError("pymzml 未读取到 MS1 光谱")

            self._ms_exp = specs
            self._mzml_ready = True
            self.status.showMessage("mzML 已就绪（pymzml）", 4000)
            QtCore.QTimer.singleShot(0, self._try_plot_selected_row)
        except Exception as e:
            self.status.showMessage(f"mzML 加载失败（pymzml）：{e}", 10000)
            QtWidgets.QMessageBox.warning(self, "EIC 预览不可用", f"mzML 加载失败（pymzml）：\n{e}")
            self._ms_exp = None
            self._mzml_ready = False

    def _try_plot_selected_row(self) -> None:
        if self._feature_df_view.empty:
            return
        sm = self.table.selectionModel()
        if sm is None:
            return
        indexes = sm.selectedRows()
        if not indexes:
            self.table.selectRow(0)
            indexes = sm.selectedRows()
            if not indexes:
                return
        self._request_plot_for_row(int(indexes[0].row()))

    def _on_canvas_context_menu(self, pos: QtCore.QPoint) -> None:
        if self.canvas is None:
            return
        menu = QtWidgets.QMenu(self)
        act_export = menu.addAction("导出当前图像…")
        act_export.setEnabled(self.fig is not None and self._last_feature_id is not None)
        chosen = menu.exec(self.canvas.mapToGlobal(pos))
        if chosen == act_export:
            self._export_current_plot()

    def _export_current_plot(self) -> None:
        if self.fig is None or self.canvas is None:
            self.status.showMessage("当前无可导出的图像", 5000)
            return

        default_name = f"{self._last_feature_id or 'eic'}.png"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存当前 EIC 图像",
            str(Path.cwd() / default_name),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)",
        )
        if not out_path:
            return

        self.fig.savefig(out_path, dpi=150)
        self.status.showMessage(f"图像已导出：{out_path}", 6000)

    def _resolve_mzml_for_attribute_export(self) -> Optional[Path]:
        if self.input_mode_combo.currentData() == "single":
            p = Path(self.mzml_path_edit.text().strip()) if self.mzml_path_edit.text().strip() else None
            if p is not None and p.exists():
                return p
        else:
            d = Path(self.mzml_dir_edit.text().strip()) if self.mzml_dir_edit.text().strip() else None
            if d is not None and d.exists() and d.is_dir():
                files = sorted(d.glob("*.mzML"))
                if files:
                    return files[0]

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择用于峰属性计算的 mzML", str(Path.cwd()), "mzML (*.mzML)")
        if not path:
            return None
        return Path(path)

    def _export_all_peak_attributes(self) -> None:
        if self._feature_df_raw is None or self._feature_df_raw.empty:
            self.status.showMessage("请先运行并加载当前算法特征表", 6000)
            return

        mzml_path = self._resolve_mzml_for_attribute_export()
        if mzml_path is None or not mzml_path.exists():
            self.status.showMessage("mzML 无效，无法计算峰属性", 6000)
            return

        algo = self.algo_combo.currentText().strip().lower()
        df = standardize_rt_columns_for_display(self._feature_df_raw.copy(), algo)
        if "RT" not in df.columns:
            self.status.showMessage("当前特征表缺少 RT 列", 6000)
            return

        base_dir = Path(self.results_dir_edit.text().strip()) if self.results_dir_edit.text().strip() else Path.cwd()
        out_default = base_dir / f"peak_attributes_all_{algo}.csv"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存峰属性（全部特征）", str(out_default), "CSV (*.csv)")
        if not out_path:
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            # Fixed by design (reference SoftwareBenchmarking/PeakAttributeCalculator.r)
            fixed_mz_tol_da = 0.01
            fixed_rt_tol_sec = 30.0
            out = compute_peak_attributes(
                df[[c for c in ["Feature_ID", "mz", "RT"] if c in df.columns]],
                mzml_path,
                mz_tolerance=fixed_mz_tol_da,
                tolerance_unit="Da",
                method="nearest",
                rt_tol_sec=fixed_rt_tol_sec,
            )
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))
            self.status.showMessage(f"峰属性导出失败：{e}", 10000)
            return
        finally:
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass

        out.to_csv(out_path, index=False)
        self.status.showMessage(f"峰属性已导出：{out_path}", 8000)

    def _export_missing_peak_attributes(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择多个算法特征峰表",
            str(Path.cwd()),
            "Table (*.csv *.xlsx *.xls)",
        )
        if not paths or len(paths) < 2:
            self.status.showMessage("至少需要选择两个算法特征表", 6000)
            return

        mzml_path = self._resolve_mzml_for_attribute_export()
        if mzml_path is None or not mzml_path.exists():
            self.status.showMessage("mzML 无效，无法计算峰属性", 6000)
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录", str(Path.cwd()))
        if not out_dir:
            return
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        try:
            # Fixed by design (reference SoftwareBenchmarking scripts)
            fixed_align_mz_tol_da = 0.01
            fixed_align_rt_tol_sec = 10.0
            fixed_attr_mz_tol_da = 0.01
            fixed_attr_rt_tol_sec = 30.0

            table_map: dict[str, pd.DataFrame] = {}
            for p_str in paths:
                p = Path(p_str)
                algo = p.parent.name.lower()
                if algo in table_map:
                    algo = p.stem
                table_map[algo] = load_and_standardize_table(p, algo=algo)

            aligned = align_feature_tables(
                table_map,
                mz_tol_da=fixed_align_mz_tol_da,
                rt_tol_sec=fixed_align_rt_tol_sec,
            )
            aligned_path = out_dir_p / "aligned_features.csv"
            aligned.to_csv(aligned_path, index=False)

            for algo in table_map.keys():
                missing = missing_features_for_algo(aligned, algo, list(table_map.keys()))
                if missing.empty:
                    continue
                attrs = compute_peak_attributes(
                    missing[["Aligned_ID", "mz", "RT"]],
                    mzml_path,
                    mz_tolerance=fixed_attr_mz_tol_da,
                    tolerance_unit="Da",
                    method="nearest",
                    rt_tol_sec=fixed_attr_rt_tol_sec,
                )
                attrs.to_csv(out_dir_p / f"missing_peak_attributes_{algo}.csv", index=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))
            self.status.showMessage(f"多算法峰属性导出失败：{e}", 10000)
            return

        self.status.showMessage(f"多算法峰属性导出完成：{out_dir_p}", 10000)

    def _export_batch_eic(self, auto: bool = False) -> None:
        if (not self._mzml_ready) or self._mzml_path is None:
            self.status.showMessage("请先加载可用 mzML 后再批量导出", 7000)
            return
        if self._ms_exp is None:
            self.status.showMessage("mzML 未就绪，请先运行并加载", 7000)
            return
        if self._feature_df_view.empty:
            self.status.showMessage("当前没有可导出的特征", 5000)
            return

        results_base_text = self.results_dir_edit.text().strip()
        if results_base_text:
            base_dir = normalize_results_base_dir(Path(results_base_text), self.algo_combo.currentText().strip().lower())
        else:
            base_dir = Path.cwd()
        out_dir = (base_dir / "eic_export").resolve()

        if self.export_scope_combo.currentData() == "all":
            max_n = len(self._feature_df_view)
        else:
            max_n = min(int(self.export_topn_spin.value()), len(self._feature_df_view))

        if (not auto) and max_n >= 1000:
            btn = QtWidgets.QMessageBox.question(
                self,
                "确认导出",
                f"即将导出 {max_n} 张EIC图，是否继续？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if btn != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if self._batch_thread is not None and self._batch_thread.isRunning():
            self.status.showMessage("已有批量导出任务在运行", 5000)
            return

        class BatchExportWorker(QtCore.QObject):
            finished = QtCore.Signal(int, str)
            failed = QtCore.Signal(str)

            def __init__(self, df: pd.DataFrame, mzml_path: Path, out_dir: Path, ppm: float, max_features: int):
                super().__init__()
                self.df = df
                self.mzml_path = mzml_path
                self.out_dir = out_dir
                self.ppm = ppm
                self.max_features = max_features

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    from lipidbench.eic.extract_eic_pyopenms import build as build_eic

                    eic_cfg = self.parent()._config_defaults.get("eic_export", {}) if self.parent() is not None else {}
                    method = str(eic_cfg.get("method", "nearest")).strip().lower()
                    unit = str(eic_cfg.get("unit", "ppm")).strip()
                    sigma = float(eic_cfg.get("smooth_sigma", 0.0))
                    # 固定图像参数（模型输入一致性）：不对用户暴露可配置项
                    window_min = 2.0
                    image_width_px = 400
                    image_height_px = 300
                    image_dpi = 100

                    df_info = self.df.copy()
                    if "RT" not in df_info.columns:
                        raise ValueError("批量导出需要 RT 列")
                    df_info = df_info[["Feature_ID", "mz", "RT"]].dropna(subset=["mz", "RT"]).head(self.max_features)

                    eic_args = SimpleNamespace(
                        processes_number=1,
                        method=("window_sum" if method == "window_sum" else "nearest"),
                        unit=("Da" if unit.lower() == "da" else "ppm"),
                        tolerance=float(self.ppm),
                        images_path=str(self.out_dir),
                        smooth_sigma=sigma,
                        window_min=window_min,
                        image_width_px=image_width_px,
                        image_height_px=image_height_px,
                        image_dpi=image_dpi,
                    )

                    _ = build_eic(paths=[self.mzml_path], info=df_info, plot=True, args=eic_args)
                    count = len(df_info)
                except Exception as e:
                    self.failed.emit(str(e))
                    return
                self.finished.emit(int(count), str(self.out_dir))

        self.export_batch_btn.setEnabled(False)
        self.status.showMessage("批量导出中…")
        self._batch_thread = QtCore.QThread(self)
        self._batch_worker = BatchExportWorker(
            df=self._feature_df_view.copy(),
            mzml_path=self._mzml_path,
            out_dir=out_dir,
            ppm=float(self.ppm_spin.value()),
            max_features=int(max_n),
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.finished.connect(self._on_batch_export_finished)
        self._batch_worker.failed.connect(self._on_batch_export_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_thread.start()

    def _on_batch_export_finished(self, count: int, out_dir: str) -> None:
        self.export_batch_btn.setEnabled(True)
        self.status.showMessage(f"批量导出完成：{count} 张 -> {out_dir}", 8000)

    def _on_batch_export_failed(self, msg: str) -> None:
        self.export_batch_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "批量导出失败", msg)
        self.status.showMessage(f"批量导出失败：{msg}", 10000)

    def _on_row_selected(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        self._request_plot_for_row(int(indexes[0].row()))

    def _request_plot_for_row(self, row: int) -> None:
        if (not self._mzml_ready) or self._mzml_path is None or self._ms_exp is None or self.canvas is None or self.ax is None:
            if self._feature_df_view is not None and not self._feature_df_view.empty:
                self.status.showMessage("EIC 预览待就绪：请确认 mzML 已加载完成", 3000)
            return
        if self._feature_df_view.empty:
            return
        if row < 0 or row >= len(self._feature_df_view):
            return

        rec = self._feature_df_view.iloc[row]

        mz = float(rec["mz"]) if pd.notna(rec["mz"]) else None
        rt_val = rec.get("RT", pd.NA)
        rtmin_val = rec.get("RTmin", pd.NA)
        rtmax_val = rec.get("RTmax", pd.NA)
        rt_center = float(rt_val) if pd.notna(rt_val) else None
        rtmin = float(rtmin_val) if pd.notna(rtmin_val) else None
        rtmax = float(rtmax_val) if pd.notna(rtmax_val) else None
        feature_id = str(rec["Feature_ID"]) if pd.notna(rec["Feature_ID"]) else f"row{row}"

        if mz is None:
            return

        if rt_center is None and rtmin is not None and rtmax is not None:
            rt_center = (rtmin + rtmax) / 2.0
        if rt_center is None:
            self.status.showMessage("该特征缺少 RT，无法按固定窗口绘图", 5000)
            return

        ppm = float(self.ppm_spin.value())
        half_window = 1.0  # fixed 2-min window for stable preview / downstream modeling
        rt_min_limit = rt_center - half_window
        rt_max_limit = rt_center + half_window

        payload = (float(mz), ppm, rt_min_limit, rt_max_limit, feature_id)
        if self._trace_busy:
            self._trace_pending = payload
            return
        self._start_trace_job(*payload)

    def _start_trace_job(
        self,
        mz: float,
        ppm: float,
        rt_min_limit: Optional[float],
        rt_max_limit: Optional[float],
        feature_id: str,
    ) -> None:
        self._trace_busy = True
        self._trace_job_id += 1
        job_id = self._trace_job_id

        class TraceWorker(QtCore.QObject):
            finished = QtCore.Signal(int, object, object, str, object, object)
            failed = QtCore.Signal(int, str)

            def __init__(self, exp: object, mz: float, ppm: float, rt_min_limit: Optional[float], rt_max_limit: Optional[float], feature_id: str, job_id: int):
                super().__init__()
                self.exp = exp
                self.mz = mz
                self.ppm = ppm
                self.rt_min_limit = rt_min_limit
                self.rt_max_limit = rt_max_limit
                self.feature_id = feature_id
                self.job_id = job_id

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    from lipidbench.eic.extract_eic_pyopenms import extract_eic_trace

                    trace = extract_eic_trace(
                        self.exp,
                        target_mz=self.mz,
                        tolerance=self.ppm,
                        unit="ppm",
                        rt_min_limit=self.rt_min_limit,
                        rt_max_limit=self.rt_max_limit,
                        method="nearest",
                        ms_level=1,
                    )
                except Exception as e:
                    self.failed.emit(self.job_id, str(e))
                    return
                self.finished.emit(
                    self.job_id,
                    trace.rt_min,
                    trace.intensity,
                    self.feature_id,
                    self.rt_min_limit,
                    self.rt_max_limit,
                )

        self._trace_thread = QtCore.QThread(self)
        self._trace_worker = TraceWorker(
            exp=self._ms_exp,
            mz=mz,
            ppm=ppm,
            rt_min_limit=rt_min_limit,
            rt_max_limit=rt_max_limit,
            feature_id=feature_id,
            job_id=job_id,
        )
        self._trace_worker.moveToThread(self._trace_thread)
        self._trace_thread.started.connect(self._trace_worker.run)
        self._trace_worker.finished.connect(self._on_trace_ready)
        self._trace_worker.failed.connect(self._on_trace_failed)
        self._trace_worker.finished.connect(self._trace_thread.quit)
        self._trace_worker.failed.connect(self._trace_thread.quit)
        self._trace_thread.finished.connect(self._trace_worker.deleteLater)
        self._trace_thread.finished.connect(self._trace_thread.deleteLater)
        self._trace_thread.start()

    def _on_trace_ready(
        self,
        job_id: int,
        rt_values: object,
        int_values: object,
        feature_id: str,
        rt_min_limit: object,
        rt_max_limit: object,
    ) -> None:
        self._trace_busy = False
        if job_id != self._trace_job_id:
            if self._trace_pending is not None:
                pending = self._trace_pending
                self._trace_pending = None
                self._start_trace_job(*pending)
            return
        if self.ax is None or self.fig is None or self.canvas is None:
            if self._trace_pending is not None:
                pending = self._trace_pending
                self._trace_pending = None
                self._start_trace_job(*pending)
            return

        self.ax.clear()
        self.ax.plot(rt_values, int_values, linewidth=1.0)
        if rt_min_limit is not None and rt_max_limit is not None:
            self.ax.set_xlim(float(rt_min_limit), float(rt_max_limit))
        self.ax.set_xlabel("RT (min)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title(feature_id)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._last_feature_id = feature_id
        self.export_plot_btn.setEnabled(True)
        if self._trace_pending is not None:
            pending = self._trace_pending
            self._trace_pending = None
            self._start_trace_job(*pending)

    def _on_trace_failed(self, job_id: int, msg: str) -> None:
        self._trace_busy = False
        if job_id != self._trace_job_id:
            if self._trace_pending is not None:
                pending = self._trace_pending
                self._trace_pending = None
                self._start_trace_job(*pending)
            return
        self.status.showMessage(f"EIC 提取失败：{msg}", 10000)
        if self._trace_pending is not None:
            pending = self._trace_pending
            self._trace_pending = None
            self._start_trace_job(*pending)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
