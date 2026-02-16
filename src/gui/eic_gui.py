from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import copy
import shutil
import tempfile
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

# Ensure `src` is importable when running from project root or executing this file directly.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gui.pandas_table_model import PandasTableModel

from utils.feature_table_io import (
    find_feature_table,
    load_feature_table,
    normalize_results_base_dir,
    standardize_rt_columns_for_display,
    suggest_peak_column,
)


def _pyopenms_supported() -> bool:
    # pyopenms binaries on Windows typically lag behind latest Python.
    return sys.version_info < (3, 13)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LipidBench EIC Viewer")
        self.resize(1200, 700)

        self._feature_df_raw: Optional[pd.DataFrame] = None
        self._feature_df_view: pd.DataFrame = pd.DataFrame(columns=["Feature_ID", "mz", "RT", "RTmin", "RTmax", "PeakArea"])

        self._mzml_path: Optional[Path] = None
        self._exp = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        header_box = QtWidgets.QGroupBox("运行")
        form = QtWidgets.QFormLayout(header_box)
        form.setContentsMargins(10, 8, 10, 8)

        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(["asari", "pyopenms", "xcms", "msdial"])

        self.mzml_path_edit = QtWidgets.QLineEdit()
        self.mzml_path_edit.setPlaceholderText("选择单个 mzML 文件")
        self.mzml_browse_btn = QtWidgets.QPushButton("选择…")
        self.mzml_browse_btn.clicked.connect(self._browse_mzml)
        mzml_row = QtWidgets.QHBoxLayout()
        mzml_row.addWidget(self.mzml_path_edit, 1)
        mzml_row.addWidget(self.mzml_browse_btn)
        mzml_row_w = QtWidgets.QWidget()
        mzml_row_w.setLayout(mzml_row)

        self.results_dir_edit = QtWidgets.QLineEdit()
        self.results_dir_edit.setPlaceholderText("选择导出结果目录（会创建 xcms/pyopenms/asari/msdial 子目录）")
        self.results_browse_btn = QtWidgets.QPushButton("选择…")
        self.results_browse_btn.clicked.connect(self._browse_results_dir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.results_dir_edit, 1)
        out_row.addWidget(self.results_browse_btn)
        out_row_w = QtWidgets.QWidget()
        out_row_w.setLayout(out_row)

        self.ppm_spin = QtWidgets.QDoubleSpinBox()
        self.ppm_spin.setRange(0.1, 200.0)
        self.ppm_spin.setDecimals(2)
        self.ppm_spin.setValue(10.0)

        self.run_btn = QtWidgets.QPushButton("运行并加载")
        self.run_btn.clicked.connect(self._run_and_load)
        self.run_btn.setDefault(True)

        form.addRow("算法", self.algo_combo)
        form.addRow("mzML", mzml_row_w)
        form.addRow("输出目录", out_row_w)
        form.addRow("EIC ppm", self.ppm_spin)
        form.addRow("", self.run_btn)

        # Main splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.table = QtWidgets.QTableView()
        self.model = PandasTableModel(self._feature_df_view)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)

        table_wrap = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_wrap)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QtWidgets.QLabel("特征列表"))
        table_layout.addWidget(self.table)

        # Plot area (matplotlib)
        plot_wrap = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_wrap)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
        except Exception as e:
            msg = QtWidgets.QLabel(f"matplotlib Qt backend 不可用：{e}")
            msg.setWordWrap(True)
            plot_layout.addWidget(msg)
            self.canvas = None
            self.fig = None
            self.ax = None
        else:
            self.fig = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            plot_layout.addWidget(QtWidgets.QLabel("EIC 图"))
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

    def _browse_results_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择结果目录（包含 xcms/pyopenms/asari/msdial 子目录）", str(Path.cwd()))
        if path:
            self.results_dir_edit.setText(path)

    def _browse_mzml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 mzML", str(Path.cwd()), "mzML (*.mzML)")
        if path:
            self.mzml_path_edit.setText(path)

    def _run_and_load(self) -> None:
        algo = self.algo_combo.currentText().strip().lower()
        mzml_path = Path(self.mzml_path_edit.text().strip()) if self.mzml_path_edit.text().strip() else None
        results_dir = Path(self.results_dir_edit.text().strip()) if self.results_dir_edit.text().strip() else None

        if mzml_path is None or not mzml_path.exists():
            self.status.showMessage("mzML 路径无效", 5000)
            return
        if results_dir is None:
            self.status.showMessage("输出目录无效", 5000)
            return
        results_dir = normalize_results_base_dir(results_dir, algo)
        results_dir.mkdir(parents=True, exist_ok=True)

        if algo == "msdial":
            # MS-DIAL 本身无法在这里直接运行；这里提供“处理导出的表格”能力。
            xlsx, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 MS-DIAL 导出的 xlsx", str(results_dir), "Excel (*.xlsx *.xls)")
            if not xlsx:
                return
            self._start_worker(algo=algo, mzml_path=mzml_path, results_dir=results_dir, msdial_xlsx=Path(xlsx))
            return

        if algo == "pyopenms" and not _pyopenms_supported():
            QtWidgets.QMessageBox.critical(
                self,
                "pyopenms 无法导入",
                "检测到你正在使用 Python 3.13。\n\n"
                "pyopenms 在 Windows 上通常需要较旧的 Python 版本（常见为 3.9/3.10/3.11）。\n"
                "这会导致 ‘DLL load failed’ 之类的导入错误。\n\n"
                "解决办法：\n"
                "1) 使用 `py -3.9` 或 `py -3.11` 创建虚拟环境并切换 VS Code 解释器；\n"
                "2) 在新环境里重新安装依赖与 pyopenms。",
            )
            self.status.showMessage("pyopenms 不支持当前 Python 版本", 8000)
            return

        self._start_worker(algo=algo, mzml_path=mzml_path, results_dir=results_dir, msdial_xlsx=None)

    def _start_worker(self, *, algo: str, mzml_path: Path, results_dir: Path, msdial_xlsx: Optional[Path]) -> None:
        self.run_btn.setEnabled(False)
        self.status.showMessage("运行算法中…")

        class Worker(QtCore.QObject):
            finished = QtCore.Signal(str, object)  # algorithm, df
            failed = QtCore.Signal(str)

            def __init__(self, algo: str, mzml_path: Path, results_dir: Path, msdial_xlsx: Optional[Path]):
                super().__init__()
                self.algo = algo
                self.mzml_path = mzml_path
                self.results_dir = results_dir
                self.msdial_xlsx = msdial_xlsx

            @QtCore.Slot()
            def run(self) -> None:
                try:
                    df = self._run_impl()
                except Exception as e:
                    self.failed.emit(str(e))
                    return
                self.finished.emit(self.algo, df)

            def _run_impl(self) -> pd.DataFrame:
                # Load config (parameters) and run single-file by copying mzML into a temp folder.
                from utils.config_io import load_config

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
                    from utils.data_io import load_msdial_results

                    load_msdial_results(self.msdial_xlsx, out_file)
                    feature_path = out_file
                    return load_feature_table(feature_path, algo)

                tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_mzml_", dir=str(results_dir)))
                shutil.copy2(self.mzml_path, tmp_dir / self.mzml_path.name)

                if algo == "xcms":
                    out_dir = results_dir / "xcms"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / "xcms_features.csv"
                    from runners.run_xcms import extract_xcms_params, run_xcms
                    from utils.data_io import load_xcms_results

                    params = extract_xcms_params(config)
                    run_xcms(input_dir=tmp_dir, output_file=out_file, **params)
                    load_xcms_results(out_file)
                    feature_path = out_file
                    return load_feature_table(feature_path, algo)

                if algo == "pyopenms":
                    out_dir = results_dir / "pyopenms"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / "pyopenms_features.csv"
                    from runners.run_pyopenms import extract_pyopenms_params, run_pyopenms
                    from utils.data_io import load_pyopenms_results

                    params = extract_pyopenms_params(config)
                    run_pyopenms(input_dir=tmp_dir, output_file=out_file, **params)
                    load_pyopenms_results(out_file, input_dir=tmp_dir, **params)
                    feature_path = out_file
                    return load_feature_table(feature_path, algo)

                if algo == "asari":
                    # Run asari CLI and post-process into CSVs.
                    cfg = copy.deepcopy(config)
                    cfg.setdefault("paths", {})
                    cfg["paths"]["input_dir"] = str(tmp_dir)
                    cfg["paths"]["asari_output"] = str((results_dir / "asari").resolve())
                    from runners.run_asari import run_asari_pipeline

                    run_asari_pipeline(cfg)
                    feature_path = find_feature_table(results_dir, algo)
                    return load_feature_table(feature_path, algo)

                raise ValueError(f"Unknown algorithm: {algo}")

        self._worker_thread = QtCore.QThread(self)
        self._worker = Worker(algo, mzml_path, results_dir, msdial_xlsx)
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
        self.status.showMessage(f"运行失败：{msg}", 12000)

    def _on_worker_finished(self, algo: str, df: pd.DataFrame) -> None:
        self.run_btn.setEnabled(True)
        self._feature_df_raw = df
        self._recompute_view()

        # Load mzML for EIC plotting
        mzml_path = Path(self.mzml_path_edit.text().strip())
        if mzml_path.exists():
            self._mzml_path = mzml_path
            self._load_mzml()

        self.status.showMessage("运行完成并加载", 5000)

    def _recompute_view(self) -> None:
        if self._feature_df_raw is None:
            return

        algo = self.algo_combo.currentText().strip().lower()
        df = standardize_rt_columns_for_display(self._feature_df_raw, algo)

        peak_col = suggest_peak_column(self._feature_df_raw, algo)
        if peak_col is not None and peak_col in df.columns:
            peak_area = pd.to_numeric(df[peak_col], errors="coerce")
        else:
            peak_area = pd.Series([pd.NA] * len(df), index=df.index)

        view = pd.DataFrame(
            {
                "Feature_ID": df.get("Feature_ID"),
                "mz": df.get("mz"),
                "RT": df.get("RT"),
                "RTmin": df.get("RTmin"),
                "RTmax": df.get("RTmax"),
                "PeakArea": peak_area,
            }
        )

        self._feature_df_view = view
        self.model.set_dataframe(self._feature_df_view)
        self.table.resizeColumnsToContents()

    def _load_mzml(self) -> None:
        if self._mzml_path is None:
            return

        try:
            import pyopenms as oms
        except Exception as e:
            self.status.showMessage(f"pyopenms 未安装/不可用：{e}", 8000)
            return

        self.status.showMessage("加载 mzML（可能较慢）…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            exp = oms.MSExperiment()
            oms.MzMLFile().load(str(self._mzml_path), exp)
            self._exp = exp
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_row_selected(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
        if self._exp is None or self.canvas is None or self.ax is None:
            return
        if self._feature_df_view.empty:
            return

        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        row = int(indexes[0].row())
        rec = self._feature_df_view.iloc[row]

        mz = float(rec["mz"]) if pd.notna(rec["mz"]) else None
        rtmin = float(rec["RTmin"]) if pd.notna(rec["RTmin"]) else None
        rtmax = float(rec["RTmax"]) if pd.notna(rec["RTmax"]) else None
        feature_id = str(rec["Feature_ID"]) if pd.notna(rec["Feature_ID"]) else f"row{row}"

        if mz is None:
            return

        from eic.extract_eic_pyopenms import extract_eic_nearest_ppm

        ppm = float(self.ppm_spin.value())

        # Crop trace to the feature window (with a small padding) for faster and clearer display.
        pad = 0.2
        rt_min_limit = (rtmin - pad) if rtmin is not None else None
        rt_max_limit = (rtmax + pad) if rtmax is not None else None

        trace = extract_eic_nearest_ppm(
            self._exp,
            target_mz=mz,
            ppm=ppm,
            rt_min_limit=rt_min_limit,
            rt_max_limit=rt_max_limit,
            ms_level=1,
        )

        self.ax.clear()
        self.ax.plot(trace.rt_min, trace.intensity, linewidth=1.0)
        self.ax.set_xlabel("RT (min)")
        self.ax.set_ylabel("Intensity")
        title = feature_id
        # Keep plot title minimal (no extra bracket/pipe content).
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
