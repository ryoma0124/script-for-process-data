import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize_scalar
from scipy import stats
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading


class WAXSBackgroundSubtractionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WAXS Background Subtraction")
        self.root.geometry("900x700")

        # 変数の初期化
        self.bg_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_prefix = tk.StringVar(value="sub_")
        self.q_min = tk.DoubleVar(value=3.5)
        self.q_max = tk.DoubleVar(value=4.5)
        self.use_manual_coef = tk.BooleanVar(value=False)
        self.manual_coef = tk.DoubleVar(value=1.0)
        self.processing = False
        self.coefficients = {}

        # サンプルファイルリスト
        self.sample_files = []

        # UIの作成
        self.create_ui()

    def create_ui(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # サンプルファイル選択エリア
        sample_frame = ttk.LabelFrame(main_frame, text="Sample Files")
        sample_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        sample_buttons_frame = ttk.Frame(sample_frame)
        sample_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(sample_buttons_frame, text="Add Files", command=self.add_sample_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(sample_buttons_frame, text="Clear All", command=self.clear_sample_files).pack(side=tk.LEFT, padx=5)

        # サンプルファイルリストボックス
        listbox_frame = ttk.Frame(sample_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.sample_listbox = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, height=6)
        self.sample_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        listbox_scrollbar = ttk.Scrollbar(listbox_frame, command=self.sample_listbox.yview)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sample_listbox.config(yscrollcommand=listbox_scrollbar.set)

        # バックグラウンドファイル
        bg_frame = ttk.LabelFrame(main_frame, text="Background File")
        bg_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(bg_frame, textvariable=self.bg_file, width=70).pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        ttk.Button(bg_frame, text="Browse", command=self.browse_bg_file).pack(side=tk.RIGHT, padx=5, pady=5)

        # 出力ディレクトリ
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory")
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(output_frame, textvariable=self.output_dir, width=70).pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.RIGHT, padx=5, pady=5)

        # 出力ファイル名プレフィックス
        prefix_frame = ttk.LabelFrame(main_frame, text="Output File Prefix")
        prefix_frame.pack(fill=tk.X, pady=5)

        ttk.Label(prefix_frame, text="ファイル名のプレフィックス:").pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Entry(prefix_frame, textvariable=self.output_prefix, width=20).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(prefix_frame, text="例: [prefix]_filename.dat").pack(side=tk.LEFT, padx=5, pady=5)

        # パラメータ設定
        param_frame = ttk.LabelFrame(main_frame, text="Parameters")
        param_frame.pack(fill=tk.X, pady=5)

        ttk.Label(param_frame, text="Q range for optimization:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(param_frame, text="Min:").grid(row=0, column=1, padx=5, pady=5, sticky=tk.E)
        self.q_min_entry = ttk.Entry(param_frame, textvariable=self.q_min, width=6)
        self.q_min_entry.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(param_frame, text="Max:").grid(row=0, column=3, padx=5, pady=5, sticky=tk.E)
        self.q_max_entry = ttk.Entry(param_frame, textvariable=self.q_max, width=6)
        self.q_max_entry.grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)

        # 手動係数モード
        self.manual_check = ttk.Checkbutton(
            param_frame,
            text="Use manual coefficient (skip auto-optimization)",
            variable=self.use_manual_coef,
            command=self.on_manual_mode_changed,
        )
        self.manual_check.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(param_frame, text="Coefficient:").grid(row=1, column=3, padx=5, pady=5, sticky=tk.E)
        self.manual_coef_entry = ttk.Entry(param_frame, textvariable=self.manual_coef, width=10)
        self.manual_coef_entry.grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)

        # 初期状態の反映
        self.on_manual_mode_changed()

        # ボタンフレーム
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.process_button = ttk.Button(button_frame, text="Process Files", command=self.start_processing)
        self.process_button.pack(side=tk.RIGHT, padx=5)

        ttk.Button(button_frame, text="Preview Selected", command=self.preview_selected).pack(side=tk.RIGHT, padx=5)

        # プログレスバー
        self.progress_frame = ttk.LabelFrame(main_frame, text="Progress")
        self.progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W, padx=5)

        # ログ表示
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # グラフエリア
        graph_frame = ttk.LabelFrame(main_frame, text="Preview")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ツールバー
        toolbar_frame = ttk.Frame(graph_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def add_sample_files(self):
        filenames = filedialog.askopenfilenames(filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        for filename in filenames:
            if filename not in self.sample_files:
                self.sample_files.append(filename)
                self.sample_listbox.insert(tk.END, os.path.basename(filename))

    def clear_sample_files(self):
        self.sample_files = []
        self.sample_listbox.delete(0, tk.END)

    def browse_bg_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
        if filename:
            self.bg_file.set(filename)

    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def on_manual_mode_changed(self):
        """手動係数モードの切替時にQレンジ入力の有効/無効を切り替える"""
        if self.use_manual_coef.get():
            # 手動モード: Qレンジは不要なので無効化、係数入力を有効化
            self.q_min_entry.config(state=tk.DISABLED)
            self.q_max_entry.config(state=tk.DISABLED)
            self.manual_coef_entry.config(state=tk.NORMAL)
        else:
            # 自動モード: Qレンジを有効化、係数入力を無効化
            self.q_min_entry.config(state=tk.NORMAL)
            self.q_max_entry.config(state=tk.NORMAL)
            self.manual_coef_entry.config(state=tk.DISABLED)

    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            data_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    continue
                # '#' なしのヘッダー行をスキップ (例: "Q (nm^-1)\tIntensity")
                if line.startswith('Q ') or line.startswith('Frame '):
                    continue
                data_lines.append(line)

            # データがない場合はエラー
            if not data_lines:
                raise ValueError(f"No valid data found in file: {filename}")

            # データの解析を試みる
            try:
                # タブ区切りを試す
                if '\t' in data_lines[0]:
                    data = []
                    for line in data_lines:
                        parts = line.split('\t')
                        if len(parts) >= 2:  # 少なくとも2つの値が必要
                            data.append([float(parts[0]), float(parts[1])])

                # カンマ区切りを試す
                elif ',' in data_lines[0]:
                    data = []
                    for line in data_lines:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            data.append([float(parts[0]), float(parts[1])])

                # スペース区切りを試す
                else:
                    data = []
                    for line in data_lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            data.append([float(parts[0]), float(parts[1])])

                # データを NumPy 配列に変換
                data = np.array(data)

            except Exception as e:
                self.log(f"Error parsing data: {str(e)}")
                self.log(f"First data line: {data_lines[0]}")
                raise ValueError(f"Failed to parse data in file: {filename}")

        return data[:,0], data[:,1]

    def correct_background(self, sample_filename, bg_filename, draw_plot=True):
        try:
            q_sample, I_sample = self.read_data(sample_filename)
            q_bg, I_bg = self.read_data(bg_filename)
        except Exception as e:
            self.log(f"Error reading files: {str(e)}")
            return None

        f_bg = interpolate.interp1d(q_bg, I_bg, kind='cubic', fill_value='extrapolate')
        I_bg_interp = f_bg(q_sample)

        q_min = self.q_min.get()
        q_max = self.q_max.get()

        if self.use_manual_coef.get():
            # 手動係数モード: 最適化をスキップしてユーザー指定値を使用
            optimal_coef = self.manual_coef.get()
            mode_label = "Manual"
        else:
            # 自動最適化モード: Qレンジ内の線形残差二乗和を最小化
            q_range_indices = np.where((q_sample > q_min) & (q_sample < q_max))[0]

            def objective(coef):
                I_corrected = I_sample - coef * I_bg_interp
                q_range = q_sample[q_range_indices]
                I_range = I_corrected[q_range_indices]
                slope, intercept, _, _, _ = stats.linregress(q_range, I_range)
                fitted_line = slope * q_range + intercept
                residuals = I_range - fitted_line
                return np.sum(residuals**2)

            result = minimize_scalar(objective)
            optimal_coef = result.x
            mode_label = "Auto"

        I_corrected = I_sample - optimal_coef * I_bg_interp

        # グラフの描画
        if draw_plot:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.plot(q_sample, I_sample, 'b-', label='Original')
            ax.plot(q_sample, I_bg_interp, 'g-', label='Background')
            ax.plot(q_sample, I_corrected, 'r-', label='Corrected')
            # 自動モードのときのみフィット領域をハイライト
            if not self.use_manual_coef.get():
                ax.axvspan(q_min, q_max, alpha=0.2, color='gray', label='Fit Region')
            ax.set_xlabel('q (nm^-1)')
            ax.set_ylabel('Intensity')
            ax.set_title(
                f'Sample: {os.path.basename(sample_filename)}\n'
                f'BG Coefficient: {optimal_coef:.6f} ({mode_label})'
            )
            ax.legend()
            self.canvas.draw()

        return q_sample, I_corrected, optimal_coef

    def preview_selected(self):
        """選択されたサンプルファイルのプレビューを表示"""
        selected_indices = self.sample_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Info", "Please select a sample file to preview")
            return

        bg_filename = self.bg_file.get()
        if not bg_filename or not os.path.exists(bg_filename):
            messagebox.showerror("Error", "Please specify a valid background file")
            return

        # 最初に選択されたファイルのみをプレビュー
        selected_index = selected_indices[0]
        sample_filename = self.sample_files[selected_index]

        self.update_status(f"Previewing: {os.path.basename(sample_filename)}")
        self.correct_background(sample_filename, bg_filename, draw_plot=True)
        self.update_status("Ready")

    def process_files(self):
        """複数ファイルの処理を実行"""
        output_dir = self.output_dir.get()
        bg_filename = self.bg_file.get()

        if not self.sample_files:
            messagebox.showerror("Error", "Please add sample files for processing")
            self.processing = False
            self.process_button.config(state=tk.NORMAL)
            return

        if not output_dir:
            messagebox.showerror("Error", "Please specify output directory")
            self.processing = False
            self.process_button.config(state=tk.NORMAL)
            return

        if not bg_filename or not os.path.exists(bg_filename):
            messagebox.showerror("Error", f"Background file not found: {bg_filename}")
            self.processing = False
            self.process_button.config(state=tk.NORMAL)
            return

        os.makedirs(output_dir, exist_ok=True)

        # 係数を保存するための辞書
        self.coefficients = {}

        total_files = len(self.sample_files)
        processed = 0

        self.log(f"Starting processing of {total_files} files...")
        if self.use_manual_coef.get():
            self.log(f"Mode: Manual coefficient = {self.manual_coef.get():.6f}")
        else:
            self.log(f"Mode: Auto-optimization (Q range: {self.q_min.get()} - {self.q_max.get()})")
        self.progress_var.set(0)

        for sample_filename in self.sample_files:
            if not self.processing:
                break

            if os.path.abspath(sample_filename) == os.path.abspath(bg_filename):
                self.log(f"Skipping background file: {sample_filename}")
                processed += 1
                continue

            self.update_status(f"Processing: {os.path.basename(sample_filename)}")
            self.log(f"Processing: {sample_filename}")

            result = self.correct_background(sample_filename, bg_filename, draw_plot=(processed == total_files-1))

            if result is not None:
                q_sample, I_corrected, optimal_coef = result

                # 出力ファイル名のプレフィックスを使用
                prefix = self.output_prefix.get()
                output_filename = f"{prefix}{os.path.splitext(os.path.basename(sample_filename))[0]}.dat"
                output_path = os.path.join(output_dir, output_filename)
                mode_label = "Manual" if self.use_manual_coef.get() else "Auto"
                np.savetxt(output_path, np.column_stack((q_sample, I_corrected)),
                           header=f'# q (nm^-1), I (corrected)\n# Background subtraction coefficient: {optimal_coef:.6f} ({mode_label})',
                           delimiter=',', comments='')

                self.log(f'Output: {output_path}')
                self.log(f'Background subtraction coefficient: {optimal_coef:.6f}\n')

                # ファイル名(拡張子なし)をそのままキーとして使用
                key = os.path.splitext(os.path.basename(sample_filename))[0]
                self.coefficients[key] = optimal_coef

            processed += 1
            self.progress_var.set(processed / total_files * 100)

        # 処理完了
        if self.processing:
            # 係数をテキストファイルに保存
            coef_file = os.path.join(output_dir, 'background_subtraction_coefficients.txt')
            with open(coef_file, 'w') as f:
                f.write("Sample\tBackground Subtraction Coefficient\n")
                for key in sorted(self.coefficients.keys()):
                    f.write(f"{key}\t{self.coefficients[key]:.6f}\n")

            self.log("Processing completed.")
            self.log(f"Background subtraction coefficients saved to: {coef_file}")
            self.update_status("Processing completed")
            messagebox.showinfo("Success", "File processing completed successfully")
        else:
            self.log("Processing canceled")
            self.update_status("Processing canceled")

        self.processing = False
        self.process_button.config(text="Process Files", command=self.start_processing)

    def start_processing(self):
        """処理開始"""
        if not self.processing:
            self.processing = True
            self.process_button.config(text="Cancel", command=self.cancel_processing)
            threading.Thread(target=self.process_files, daemon=True).start()

    def cancel_processing(self):
        """処理キャンセル"""
        if self.processing:
            self.processing = False
            self.update_status("Canceling...")


if __name__ == "__main__":
    root = tk.Tk()
    app = WAXSBackgroundSubtractionGUI(root)

    # コマンドライン引数がある場合に処理する
    import sys
    if len(sys.argv) > 1:
        bg_file = None
        output_dir = ""

        # 最初の引数をバックグラウンドファイルとして扱う
        if len(sys.argv) >= 2:
            bg_file = sys.argv[1]
            if os.path.exists(bg_file):
                app.bg_file.set(bg_file)

        # 出力ディレクトリの指定
        if len(sys.argv) >= 3:
            output_dir = sys.argv[2]
            app.output_dir.set(output_dir)

        # 追加の引数をサンプルファイルとして扱う
        for i in range(3, len(sys.argv)):
            sample_file = sys.argv[i]
            if os.path.exists(sample_file):
                app.sample_files.append(sample_file)
                app.sample_listbox.insert(tk.END, os.path.basename(sample_file))

    root.mainloop()
