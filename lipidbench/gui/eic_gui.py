from __future__ import annotations
import sys
import subprocess
from pathlib import Path
from typing import Optional, Any
import copy
import shutil
import tempfile
import pandas as pd
from PySide6 import QtCore, QtWidgets

from lipidbench.gui.pandas_table_model import PandasTableModel
from lipidbench.utils.feature_table_io import (
    build_feature_view_table,
    find_feature_table,
    load_feature_table,
    normalize_results_base_dir,
)


def _check_pyopenms_available() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pyopenms,sys; print(pyopenms.__version__)",
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
            f"Python: {sys.version.split()[0]}\n"
            f"解释器: {sys.executable}\n"
            f"原始错误: {e}"
        )
        return False, detail


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
        self._ms1_cache = None
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

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        header_box = QtWidgets.QGroupBox("运行")
        form = QtWidgets.QFormLayout(header_box)
        form.setContentsMargins(10, 8, 10, 8)

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
        mzml_row_w = QtWidgets.QWidget()
        mzml_row_w.setLayout(mzml_row)

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
        self.export_batch_btn.clicked.connect(self._export_batch_eic)

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

        form.addRow("算法", self.algo_combo)
        form.addRow("输入模式", self.input_mode_combo)
        form.addRow("mzML", mzml_row_w)
        form.addRow("mzML目录", self.mzml_dir_row_w)
        form.addRow("输出目录", out_row_w)
        form.addRow("MS-DIAL 表", self.msdial_row_w)
        form.addRow("参数", self.params_toggle_btn)
        form.addRow("", self.params_container)
        form.addRow("EIC ppm", self.ppm_spin)
        form.addRow("", self.run_btn)
        form.addRow("", self.export_plot_btn)
        form.addRow("", self.export_batch_btn)
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
        self.mzml_path_edit.setVisible(not is_folder)
        self.mzml_browse_btn.setVisible(not is_folder)
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
            )
            return

        if algo == "pyopenms":
            ok, detail = _check_pyopenms_available()
            if not ok:
                QtWidgets.QMessageBox.critical(self, "pyopenms 无法导入", detail)
                self.status.showMessage("pyopenms 导入失败（已弹窗显示详情）", 10000)
                return

        self._start_worker(
            algo=algo,
            input_files=input_files,
            input_mode=input_mode,
            mzml_dir=mzml_dir,
            results_dir=results_dir,
            msdial_xlsx=None,
            algo_params=algo_params,
            asari_table_preference=asari_table_preference,
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

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    df = self._run_impl()
                except Exception as e:
                    self.failed.emit(str(e))
                    return
                self.finished.emit(self.algo, df)

            def _run_impl(self) -> pd.DataFrame:
                import subprocess
                import sys
                from lipidbench.utils.config_io import load_config
                from lipidbench.utils.data_io import load_msdial_results, load_xcms_results, load_pyopenms_results
                from lipidbench.runners.run_xcms import extract_xcms_params, run_xcms
                from lipidbench.runners.run_asari import run_asari_pipeline
                from lipidbench.utils.config_io import get_base_dir

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
                    base_dir = get_base_dir()
                    cli = base_dir / "run_algo_single.py"
                    cmd = [
                        sys.executable,
                        str(cli),
                        "--algo",
                        "pyopenms",
                        "--results-dir",
                        str(results_dir),
                    ]
                    if self.input_mode == "folder" and self.mzml_dir is not None:
                        cmd += ["--mzml-dir", str(self.mzml_dir)]
                    else:
                        cmd += ["--mzml", str(self.input_files[0])]
                    if "mz_tol" in self.algo_params:
                        cmd += ["--pyopenms-ppm", str(self.algo_params["mz_tol"])]
                    if "noise" in self.algo_params:
                        cmd += ["--pyopenms-noise", str(self.algo_params["noise"])]
                    if "sn" in self.algo_params:
                        cmd += ["--pyopenms-sn", str(self.algo_params["sn"])]
                    if "min_fwhm" in self.algo_params:
                        cmd += ["--pyopenms-min-fwhm", str(self.algo_params["min_fwhm"])]
                    if "max_fwhm" in self.algo_params:
                        cmd += ["--pyopenms-max-fwhm", str(self.algo_params["max_fwhm"])]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        err = (proc.stderr or proc.stdout or "").strip()
                        raise RuntimeError(f"pyopenms 子进程运行失败：{err}")
                    out_file = results_dir / "pyopenms" / "pyopenms_features.csv"
                    if not out_file.exists():
                        raise FileNotFoundError(f"pyopenms 输出不存在: {out_file}")
                    return load_feature_table(out_file, algo)

                raise ValueError(f"Unknown algorithm: {algo}")

        self._worker_thread = QtCore.QThread(self)
        self._worker = Worker(algo, input_files, input_mode, mzml_dir, results_dir, msdial_xlsx, algo_params, asari_table_preference)
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

        self.status.showMessage("运行完成并加载", 5000)

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

        try:
            import pyopenms as oms
        except Exception as e:
            self.status.showMessage(
                f"GUI 进程内 pyopenms 不可用：{e}。",
                10000,
            )
            self._ms_exp = None
            self._ms1_cache = None
            self._mzml_ready = False
            return

        try:
            exp = oms.MSExperiment()
            oms.MzMLFile().load(str(self._mzml_path), exp)
        except Exception as e:
            self.status.showMessage(f"mzML 加载失败：{e}", 10000)
            self._ms_exp = None
            self._ms1_cache = None
            self._mzml_ready = False
            return

        try:
            from lipidbench.eic.extract_eic_pyopenms import build_ms1_cache

            cache = build_ms1_cache(exp, ms_level=1)
        except Exception as e:
            self.status.showMessage(f"MS1 缓存构建失败：{e}", 10000)
            self._ms_exp = None
            self._ms1_cache = None
            self._mzml_ready = False
            return

        self._ms_exp = exp
        self._ms1_cache = cache
        self._mzml_ready = True
        self.status.showMessage("mzML 与 MS1 缓存已就绪", 3000)

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

    def _export_batch_eic(self) -> None:
        if (not self._mzml_ready) or self._mzml_path is None:
            self.status.showMessage("请先加载可用 mzML 后再批量导出", 7000)
            return
        if self._ms_exp is None:
            self.status.showMessage("mzML 未就绪，请先运行并加载", 7000)
            return
        if self._feature_df_view.empty:
            self.status.showMessage("当前没有可导出的特征", 5000)
            return

        out_dir_text = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 EIC 导出目录", str(Path.cwd()))
        if not out_dir_text:
            return

        max_default = min(200, len(self._feature_df_view))
        max_n, ok = QtWidgets.QInputDialog.getInt(
            self,
            "批量导出",
            "导出前 N 个特征：",
            value=max_default,
            min=1,
            max=max(1, len(self._feature_df_view)),
            step=1,
        )
        if not ok:
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
                    from lipidbench.eic.export import EICImageStyle, export_eic_images_from_df

                    eic_cfg = self.parent()._config_defaults.get("eic_export", {}) if self.parent() is not None else {}
                    fixed_rt_window_min = float(eic_cfg.get("fixed_rt_window_min", 2.0))
                    style = EICImageStyle(
                        width_px=int(eic_cfg.get("width_px", 400)),
                        height_px=int(eic_cfg.get("height_px", 300)),
                        dpi=int(eic_cfg.get("dpi", 100)),
                        line_width=float(eic_cfg.get("line_width", 1.0)),
                        normalize_intensity=bool(eic_cfg.get("normalize_intensity", True)),
                        show_axes=bool(eic_cfg.get("show_axes", True)),
                        show_title=bool(eic_cfg.get("show_title", False)),
                        fixed_rt_window_min=(fixed_rt_window_min if fixed_rt_window_min > 0 else None),
                    )

                    count = export_eic_images_from_df(
                        df=self.df,
                        mzml_path=self.mzml_path,
                        out_dir=self.out_dir,
                        method="window_sum",
                        ppm=self.ppm,
                        max_features=self.max_features,
                        rt_pad_min=0.2,
                        image_style=style,
                    )
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
            out_dir=Path(out_dir_text),
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
        if (not self._mzml_ready) or self._mzml_path is None or self._ms1_cache is None or self.canvas is None or self.ax is None:
            return
        if self._feature_df_view.empty:
            return

        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        row = int(indexes[0].row())
        rec = self._feature_df_view.iloc[row]

        mz = float(rec["mz"]) if pd.notna(rec["mz"]) else None
        rtmin_val = rec.get("RTmin", pd.NA)
        rtmax_val = rec.get("RTmax", pd.NA)
        rtmin = float(rtmin_val) if pd.notna(rtmin_val) else None
        rtmax = float(rtmax_val) if pd.notna(rtmax_val) else None
        feature_id = str(rec["Feature_ID"]) if pd.notna(rec["Feature_ID"]) else f"row{row}"

        if mz is None:
            return

        ppm = float(self.ppm_spin.value())
        pad = 0.2
        rt_min_limit = (rtmin - pad) if rtmin is not None else None
        rt_max_limit = (rtmax + pad) if rtmax is not None else None

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
            finished = QtCore.Signal(int, object, object, str)
            failed = QtCore.Signal(int, str)

            def __init__(self, cache: object, mz: float, ppm: float, rt_min_limit: Optional[float], rt_max_limit: Optional[float], feature_id: str, job_id: int):
                super().__init__()
                self.cache = cache
                self.mz = mz
                self.ppm = ppm
                self.rt_min_limit = rt_min_limit
                self.rt_max_limit = rt_max_limit
                self.feature_id = feature_id
                self.job_id = job_id

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    from lipidbench.eic.extract_eic_pyopenms import extract_eic_from_cache

                    trace = extract_eic_from_cache(
                        self.cache,
                        target_mz=self.mz,
                        ppm=self.ppm,
                        rt_min_limit=self.rt_min_limit,
                        rt_max_limit=self.rt_max_limit,
                        method="nearest",
                    )
                except Exception as e:
                    self.failed.emit(self.job_id, str(e))
                    return
                self.finished.emit(self.job_id, trace.rt_min, trace.intensity, self.feature_id)

        self._trace_thread = QtCore.QThread(self)
        self._trace_worker = TraceWorker(
            cache=self._ms1_cache,
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

    def _on_trace_ready(self, job_id: int, rt_values: object, int_values: object, feature_id: str) -> None:
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
