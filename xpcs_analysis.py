#!/usr/bin/env python3
"""
XPCS Data Processing Script
============================
X-ray Photon Correlation Spectroscopy (XPCS) データ処理スクリプト

処理内容:
  1. g2(tau) データファイル (*_g2_XX.dat) の読み込み
  2. パラメータファイル (StrExpPara_*.dat) の読み込み
  3. g2(tau) の規格化: [g2 - baseline] / beta
  4. 各種プロット生成 (g2, 規格化g2, Gamma vs q, alpha vs q)
  5. Gamma ∝ q^n のべき乗フィッティング
  6. 結果のテキストファイル保存

フィッティングモデル:
  g2(q, t) = beta * exp[-2 * (Gamma * t)^alpha] + baseline
"""

import os
import re
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


# =============================================================================
# Configuration — ユーザーが編集するセクション
# =============================================================================

# 探索のルートディレクトリ（カレントディレクトリ以下を再帰探索）
ROOT_DIR = "."

# パラメータファイルの glob パターン（このファイルが存在するディレクトリを処理対象とする）
PARAM_PATTERN = "StrExpPara_*.dat"

# g2 ファイルの glob パターン
FILE_PATTERN = "*_g2_*.dat"

# 出力先ルートディレクトリ（この下にサブフォルダが自動作成される）
OUTPUT_ROOT = "./output"

# プロット保存形式 ("png" or "pdf")
PLOT_FORMAT = "png"

# プロットの DPI
PLOT_DPI = 150


# =============================================================================
# Data Loading Functions
# =============================================================================

def extract_index_from_filename(filepath):
    """
    g2 ファイル名から連番を抽出する。

    ファイル名パターン: *_g2_XX.dat
    例: sample_g2_01.dat -> 1, sample_g2_12.dat -> 12

    Parameters
    ----------
    filepath : str
        g2 データファイルのパス

    Returns
    -------
    int or None
        抽出された連番。抽出できない場合は None。
    """
    basename = os.path.basename(filepath)
    # *_g2_XX.dat のパターンから XX 部分を抽出
    match = re.search(r'_g2_(\d+)\.dat$', basename)
    if match:
        return int(match.group(1))
    else:
        warnings.warn(f"Could not extract index from filename: {basename}")
        return None


def load_g2_file(filepath):
    """
    単一の g2 データファイルを読み込む。

    ファイル構成:
      - 1行目: 'q (nm) = 0.004656, pixel = 16  (q値とピクセル数)
      - 2行目: カラム説明 (無視)
      - 3行目: # コメント行 (無視)
      - 4行目以降: tau, g2, g2_err, g2fit1, g2fit2 (タブ区切り)

    Parameters
    ----------
    filepath : str
        g2 データファイルのパス

    Returns
    -------
    dict
        {
            'q': float,          # q 値 (1行目から抽出)
            'tau': ndarray,      # 時間
            'g2': ndarray,       # g2 実測値
            'g2_err': ndarray,   # g2 エラーバー
            'fit_fixed': ndarray, # フィッティング結果 (パラメータ固定)
            'fit_free': ndarray,  # フィッティング結果 (より正確)
        }
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1行目: 'q (nm) = 0.004656, pixel = 16 から q 値を抽出
    line1 = lines[0].strip().strip("'")
    match = re.search(r'q\s*\(.*?\)\s*=\s*([\d.eE+\-]+)', line1)
    if match:
        q_value = float(match.group(1))
    else:
        # フォールバック: 数値を探す
        nums = re.findall(r'[\d.eE+\-]+', line1)
        if nums:
            q_value = float(nums[0])
        else:
            raise ValueError(f"Could not extract q value from: {lines[0].strip()}")

    # 4行目以降: データ本体 (タブ区切り、先頭3行スキップ)
    # comments パラメータで '#' と "'" で始まる行を無視
    data = np.loadtxt(filepath, skiprows=3, delimiter='\t')

    result = {
        'q': q_value,
        'tau': data[:, 0],
        'g2': data[:, 1],
        'g2_err': data[:, 2],
        'fit_fixed': data[:, 3],
        'fit_free': data[:, 4],
    }

    return result


def load_all_g2(directory, pattern):
    """
    ディレクトリ内の全 g2 ファイルを一括読み込みする。

    Parameters
    ----------
    directory : str
        データディレクトリのパス
    pattern : str
        glob パターン (例: "*_g2_*.dat")

    Returns
    -------
    dict
        {index(int): {'q', 'tau', 'g2', 'g2_err', 'fit_fixed', 'fit_free'}}
        連番をキーとした辞書
    """
    file_list = sorted(glob.glob(os.path.join(directory, pattern)))

    if not file_list:
        raise FileNotFoundError(
            f"No g2 files found matching '{pattern}' in '{directory}'"
        )

    print(f"Found {len(file_list)} g2 data file(s)")

    data_dict = {}
    for fpath in file_list:
        idx = extract_index_from_filename(fpath)
        if idx is None:
            continue
        try:
            data = load_g2_file(fpath)
            data_dict[idx] = data
            print(f"  Loaded: {os.path.basename(fpath)} -> index={idx}, q={data['q']:.6f}")
        except Exception as e:
            warnings.warn(f"Failed to load {fpath}: {e}")

    return data_dict


def find_parameter_file(directory, pattern):
    """
    ディレクトリ内からパラメータファイルを自動検索する。

    Parameters
    ----------
    directory : str
        検索対象ディレクトリ
    pattern : str
        glob パターン (例: "StrExpPara_*.dat")

    Returns
    -------
    str
        見つかったパラメータファイルのパス

    Raises
    ------
    FileNotFoundError
        パターンに一致するファイルが見つからない場合
    """
    candidates = sorted(glob.glob(os.path.join(directory, pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"No parameter file matching '{pattern}' found in '{directory}'"
        )

    if len(candidates) > 1:
        print(f"  Warning: Multiple parameter files found, using the first one:")
        for c in candidates:
            print(f"    {os.path.basename(c)}")

    selected = candidates[0]
    print(f"  Parameter file: {os.path.basename(selected)}")
    return selected


def load_parameters(filepath):
    """
    パラメータファイル (StrExpPara_*.dat) を読み込む。

    ファイル構成:
      - 1行目: ヘッダ (無視)
      - 2行目以降: 連番, q, npix, baseline, contrast(beta), Gamma,
                    alpha, baseline_err, contrast_err, gamma_err, str_err

    Parameters
    ----------
    filepath : str
        パラメータファイルのパス

    Returns
    -------
    dict
        {index(int): {
            'q': float, 'npix': float, 'baseline': float, 'beta': float,
            'gamma': float, 'alpha': float, 'baseline_err': float,
            'contrast_err': float, 'gamma_err': float, 'alpha_err': float,
        }}
    """
    data = np.loadtxt(filepath, skiprows=1)

    params = {}
    for row in data:
        idx = int(row[0])
        params[idx] = {
            'q': row[1],
            'npix': row[2],
            'baseline': row[3],
            'beta': row[4],
            'gamma': row[5],
            'alpha': row[6],
            'baseline_err': row[7],
            'contrast_err': row[8],
            'gamma_err': row[9],
            'alpha_err': row[10],
        }

    print(f"Loaded parameters for {len(params)} q-values from {os.path.basename(filepath)}")

    return params


# =============================================================================
# Data Processing Functions
# =============================================================================

def normalize_g2(g2, baseline, beta):
    """
    g2 の規格化を実行する。

    g2_norm = [g2 - baseline] / beta

    Parameters
    ----------
    g2 : ndarray
        g2(tau) 実測値
    baseline : float
        ベースライン値
    beta : float
        コントラストファクター (beta)

    Returns
    -------
    ndarray
        規格化された g2
    """
    return (g2 - baseline) / beta


def power_law(q, A, n):
    """
    べき乗則モデル: Gamma = A * q^n

    Parameters
    ----------
    q : ndarray
        散乱ベクトル
    A : float
        係数
    n : float
        べき指数

    Returns
    -------
    ndarray
        Gamma 値
    """
    return A * np.power(q, n)


def fit_gamma_vs_q(q_array, gamma_array, gamma_err=None):
    """
    Gamma vs q のべき乗フィッティングを実行する。

    モデル: Gamma = A * q^n
    両対数空間での線形フィッティング: log(Gamma) = log(A) + n * log(q)

    Parameters
    ----------
    q_array : ndarray
        q 値の配列
    gamma_array : ndarray
        Gamma 値の配列
    gamma_err : ndarray or None
        Gamma の誤差。指定時は重み付きフィッティング。

    Returns
    -------
    dict
        {
            'A': float, 'A_err': float,
            'n': float, 'n_err': float,
            'R2': float,
            'fit_values': ndarray,  # フィット曲線の値
            'q_fit': ndarray,       # フィット曲線の q 値
        }
    """
    # 対数空間でのフィッティング
    log_q = np.log10(q_array)
    log_gamma = np.log10(gamma_array)

    # 重み計算 (誤差伝播: sigma_log = sigma / (value * ln(10)))
    if gamma_err is not None:
        sigma_log = gamma_err / (gamma_array * np.log(10))
    else:
        sigma_log = None

    # 線形フィット: log(Gamma) = log(A) + n * log(q)
    def linear(x, intercept, slope):
        return intercept + slope * x

    popt, pcov = curve_fit(linear, log_q, log_gamma, sigma=sigma_log,
                           absolute_sigma=True if sigma_log is not None else False)
    perr = np.sqrt(np.diag(pcov))

    log_A = popt[0]
    n = popt[1]
    log_A_err = perr[0]
    n_err = perr[1]

    A = 10 ** log_A
    # 誤差伝播: delta_A = A * ln(10) * delta_log_A
    A_err = A * np.log(10) * log_A_err

    # R^2 計算
    log_gamma_fit = linear(log_q, *popt)
    ss_res = np.sum((log_gamma - log_gamma_fit) ** 2)
    ss_tot = np.sum((log_gamma - np.mean(log_gamma)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # フィット曲線生成 (滑らかな線)
    q_fit = np.logspace(np.log10(q_array.min()) - 0.1,
                        np.log10(q_array.max()) + 0.1, 200)
    fit_values = power_law(q_fit, A, n)

    return {
        'A': A, 'A_err': A_err,
        'n': n, 'n_err': n_err,
        'R2': R2,
        'fit_values': fit_values,
        'q_fit': q_fit,
    }


# =============================================================================
# Plot Functions
# =============================================================================

def plot_g2(data_dict, params, output_dir, fmt="png", dpi=150):
    """
    g2(tau) のプロットを生成する。
    各 q 値のデータをエラーバー付きで表示し、フィッティング曲線を重畳する。

    Parameters
    ----------
    data_dict : dict
        g2 データ辞書 (連番がキー)
    params : dict
        パラメータ辞書 (連番がキー)
    output_dir : str
        出力ディレクトリ
    fmt : str
        プロット保存形式
    dpi : int
        DPI
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # q 値でソート
    sorted_indices = sorted(data_dict.keys(),
                            key=lambda idx: params[idx]['q'] if idx in params else 0)

    cmap = plt.cm.viridis
    n_plots = len(sorted_indices)
    colors = [cmap(i / max(n_plots - 1, 1)) for i in range(n_plots)]

    for i, idx in enumerate(sorted_indices):
        d = data_dict[idx]
        q_val = d['q']
        color = colors[i]

        ax.errorbar(d['tau'], d['g2'], yerr=d['g2_err'],
                     fmt='o', markersize=3, color=color, alpha=0.7,
                     label=f"q = {q_val:.5f}", capsize=1, elinewidth=0.5)
        ax.plot(d['tau'], d['fit_free'], '-', color=color, linewidth=1.5, alpha=0.9)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$ (s)', fontsize=14)
    ax.set_ylabel(r'$g_2(\tau)$', fontsize=14)
    ax.set_title(r'$g_2(\tau)$ vs $\tau$', fontsize=16)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"g2_vs_tau.{fmt}")
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_normalized_g2(data_dict, params, output_dir, fmt="png", dpi=150):
    """
    規格化 g2 のプロットを生成する。
    [g2 - baseline] / beta を tau の関数として全 q 値を重ねて表示する。

    Parameters
    ----------
    data_dict : dict
        g2 データ辞書
    params : dict
        パラメータ辞書
    output_dir : str
        出力ディレクトリ
    fmt : str
        プロット保存形式
    dpi : int
        DPI
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    sorted_indices = sorted(data_dict.keys(),
                            key=lambda idx: params[idx]['q'] if idx in params else 0)

    cmap = plt.cm.viridis
    n_plots = len(sorted_indices)
    colors = [cmap(i / max(n_plots - 1, 1)) for i in range(n_plots)]

    for i, idx in enumerate(sorted_indices):
        if idx not in params:
            continue
        d = data_dict[idx]
        p = params[idx]
        g2_norm = normalize_g2(d['g2'], p['baseline'], p['beta'])

        ax.plot(d['tau'], g2_norm, 'o', markersize=3, color=colors[i], alpha=0.7,
                label=f"q = {d['q']:.5f}")

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$ (s)', fontsize=14)
    ax.set_ylabel(r'$[g_2(\tau) - \mathrm{baseline}] / \beta$', fontsize=14)
    ax.set_title(r'Normalized $g_2(\tau)$', fontsize=16)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"g2_normalized.{fmt}")
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_gamma_vs_q(q, gamma, gamma_err, fit_result, output_dir, fmt="png", dpi=150):
    """
    Gamma vs q の両対数プロットを生成する。

    Parameters
    ----------
    q : ndarray
        q 値
    gamma : ndarray
        Gamma 値
    gamma_err : ndarray
        Gamma の誤差
    fit_result : dict
        フィッティング結果
    output_dir : str
        出力ディレクトリ
    fmt : str
        プロット保存形式
    dpi : int
        DPI
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(q, gamma, yerr=gamma_err, fmt='o', markersize=6, color='C0',
                capsize=3, elinewidth=1, label='Data')
    ax.plot(fit_result['q_fit'], fit_result['fit_values'], '-', color='C1',
            linewidth=2,
            label=(f"Fit: $\\Gamma = A \\cdot q^n$\n"
                   f"  n = {fit_result['n']:.3f} ± {fit_result['n_err']:.3f}\n"
                   f"  A = {fit_result['A']:.4e} ± {fit_result['A_err']:.4e}\n"
                   f"  R² = {fit_result['R2']:.6f}"))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=14)
    ax.set_ylabel(r'$\Gamma$ (s$^{-1}$)', fontsize=14)
    ax.set_title(r'$\Gamma$ vs $q$', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"gamma_vs_q.{fmt}")
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_alpha_vs_q(q, alpha, alpha_err, output_dir, fmt="png", dpi=150):
    """
    alpha vs q のプロットを生成する。

    Parameters
    ----------
    q : ndarray
        q 値
    alpha : ndarray
        伸長指数 alpha
    alpha_err : ndarray
        alpha の誤差
    output_dir : str
        出力ディレクトリ
    fmt : str
        プロット保存形式
    dpi : int
        DPI
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(q, alpha, yerr=alpha_err, fmt='s', markersize=6, color='C2',
                capsize=3, elinewidth=1, label='Data')

    # alpha = 1 の参照線 (単純指数緩和)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=r'$\alpha = 1$ (simple exponential)')

    ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=14)
    ax.set_ylabel(r'$\alpha$ (stretching exponent)', fontsize=14)
    ax.set_title(r'$\alpha$ vs $q$', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"alpha_vs_q.{fmt}")
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


# =============================================================================
# Data Saving Functions
# =============================================================================

def save_normalized_g2(output_dir, data_dict, params):
    """
    規格化 g2 データをテキストファイルに保存する。
    各 q 値ごとに1ファイルを生成する。
    実測値とフィッティング結果（より正確なフィッティング）の両方を規格化する。

    出力ファイル名: normalized_g2_q{q_value}.dat
    カラム: tau, g2_normalized, g2fit_normalized

    Parameters
    ----------
    output_dir : str
        出力ディレクトリ
    data_dict : dict
        g2 データ辞書
    params : dict
        パラメータ辞書
    """
    sorted_indices = sorted(data_dict.keys(),
                            key=lambda idx: params[idx]['q'] if idx in params else 0)

    saved_count = 0
    for idx in sorted_indices:
        if idx not in params:
            continue
        d = data_dict[idx]
        p = params[idx]
        g2_norm = normalize_g2(d['g2'], p['baseline'], p['beta'])
        g2fit_norm = normalize_g2(d['fit_free'], p['baseline'], p['beta'])

        q_str = f"{p['q']:.6f}".replace('.', 'p')
        outpath = os.path.join(output_dir, f"normalized_g2_q{q_str}.dat")

        header = (
            f"Normalized g2 data\n"
            f"q = {p['q']:.6e}\n"
            f"baseline = {p['baseline']:.6e}, beta (contrast) = {p['beta']:.6e}\n"
            f"Normalization: [g2 - baseline] / beta\n"
            f"{'tau':>15s}  {'g2_normalized':>15s}  {'g2fit_normalized':>16s}"
        )
        np.savetxt(outpath, np.column_stack([d['tau'], g2_norm, g2fit_norm]),
                   header=header, fmt='%15.6e', delimiter='  ')
        saved_count += 1

    print(f"Saved {saved_count} normalized g2 file(s) to {output_dir}")


def save_gamma_vs_q(output_dir, q, gamma, gamma_err, fit_result):
    """
    Gamma vs q のデータとフィット結果をテキストファイルに保存する。

    カラム: q, Gamma, Gamma_err, Gamma_fit

    Parameters
    ----------
    output_dir : str
        出力ディレクトリ
    q : ndarray
        q 値
    gamma : ndarray
        Gamma 値
    gamma_err : ndarray
        Gamma の誤差
    fit_result : dict
        フィッティング結果
    """
    # データ点でのフィット値を計算
    gamma_fit = power_law(q, fit_result['A'], fit_result['n'])

    outpath = os.path.join(output_dir, "gamma_vs_q.dat")

    header = (
        f"Gamma vs q data with power-law fit\n"
        f"Fit model: Gamma = A * q^n\n"
        f"A = {fit_result['A']:.6e} +/- {fit_result['A_err']:.6e}\n"
        f"n = {fit_result['n']:.6f} +/- {fit_result['n_err']:.6f}\n"
        f"R^2 = {fit_result['R2']:.6f}\n"
        f"{'q':>15s}  {'Gamma':>15s}  {'Gamma_err':>15s}  {'Gamma_fit':>15s}"
    )
    np.savetxt(outpath,
               np.column_stack([q, gamma, gamma_err, gamma_fit]),
               header=header, fmt='%15.6e', delimiter='  ')
    print(f"Saved: {outpath}")


def save_alpha_vs_q(output_dir, q, alpha, alpha_err):
    """
    alpha vs q のデータをテキストファイルに保存する。

    カラム: q, alpha, alpha_err

    Parameters
    ----------
    output_dir : str
        出力ディレクトリ
    q : ndarray
        q 値
    alpha : ndarray
        伸長指数 alpha
    alpha_err : ndarray
        alpha の誤差
    """
    outpath = os.path.join(output_dir, "alpha_vs_q.dat")

    header = (
        f"Stretching exponent (alpha) vs q\n"
        f"{'q':>15s}  {'alpha':>15s}  {'alpha_err':>15s}"
    )
    np.savetxt(outpath,
               np.column_stack([q, alpha, alpha_err]),
               header=header, fmt='%15.6e', delimiter='  ')
    print(f"Saved: {outpath}")


def save_fit_summary(output_dir, fit_result):
    """
    フィッティング結果のサマリーをテキストファイルに保存する。

    Parameters
    ----------
    output_dir : str
        出力ディレクトリ
    fit_result : dict
        フィッティング結果
    """
    outpath = os.path.join(output_dir, "fit_summary.txt")

    with open(outpath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("XPCS Analysis - Fitting Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Gamma vs q Power-Law Fit\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: Gamma = A * q^n\n\n")
        f.write(f"  A = {fit_result['A']:.6e} +/- {fit_result['A_err']:.6e}\n")
        f.write(f"  n = {fit_result['n']:.6f} +/- {fit_result['n_err']:.6f}\n")
        f.write(f"  R^2 = {fit_result['R2']:.6f}\n")
        f.write("\n")

    print(f"Saved: {outpath}")


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# Directory Discovery
# =============================================================================

def find_data_directories(root_dir, param_pattern):
    """
    ルートディレクトリ以下を再帰探索し、パラメータファイルが存在する
    ディレクトリを全て検出する。

    Parameters
    ----------
    root_dir : str
        探索のルートディレクトリ
    param_pattern : str
        パラメータファイルの glob パターン (例: "StrExpPara_*.dat")

    Returns
    -------
    list of str
        パラメータファイルが見つかったディレクトリパスのリスト（ソート済み）
    """
    data_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        import fnmatch
        matches = fnmatch.filter(filenames, param_pattern)
        if matches:
            data_dirs.add(dirpath)
    return sorted(data_dirs)


# =============================================================================
# Single Directory Processing
# =============================================================================

def process_directory(data_dir, output_dir, param_pattern=PARAM_PATTERN,
                      file_pattern=FILE_PATTERN, plot_format=PLOT_FORMAT,
                      plot_dpi=PLOT_DPI):
    """単一ディレクトリのデータを処理する。"""

    os.makedirs(output_dir, exist_ok=True)

    # ----- パラメータファイル検索・読み込み -----
    print("  [Step 1] Searching for parameter file...")
    param_filepath = find_parameter_file(data_dir, param_pattern)
    params = load_parameters(param_filepath)
    print()

    # ----- 全 g2 ファイル読み込み -----
    print("  [Step 2] Loading g2 data files...")
    data_dict = load_all_g2(data_dir, file_pattern)
    print()

    # ----- 連番で紐付け・検証 -----
    print("  [Step 3] Matching g2 data with parameters...")
    matched_indices = []
    excluded_indices = []
    for idx in sorted(data_dict.keys()):
        if idx in params:
            q_data = data_dict[idx]['q']
            q_param = params[idx]['q']
            rel_diff = abs(q_data - q_param) / q_param if q_param != 0 else abs(q_data - q_param)
            if rel_diff > 0.01:
                warnings.warn(
                    f"  Index {idx}: q mismatch - data q={q_data:.6f}, param q={q_param:.6f} "
                    f"(relative diff = {rel_diff:.4f})"
                )
            if params[idx]['beta'] <= 0:
                excluded_indices.append(idx)
                print(f"    Index {idx} (q={q_param:.6f}): EXCLUDED - "
                      f"beta={params[idx]['beta']:.6f} <= 0 (likely unconverged fit)")
            else:
                matched_indices.append(idx)
        else:
            warnings.warn(f"  Index {idx}: No matching parameters found, skipping.")

    print(f"    Matched {len(matched_indices)} / {len(data_dict)} datasets"
          f" (excluded {len(excluded_indices)})")
    print()

    if len(matched_indices) == 0:
        print("  WARNING: No valid datasets found. Skipping this directory.")
        return False

    data_dict_matched = {idx: data_dict[idx] for idx in matched_indices}

    # ----- g2(tau) プロット -----
    print("  [Step 4] Generating g2(tau) plot...")
    plot_g2(data_dict_matched, params, output_dir, fmt=plot_format, dpi=plot_dpi)
    print()

    # ----- 規格化 -----
    print("  [Step 5] Normalizing g2 data: [g2 - baseline] / beta...")
    print()

    # ----- 規格化データ保存 -----
    print("  [Step 6] Saving normalized g2 data...")
    save_normalized_g2(output_dir, data_dict_matched, params)
    print()

    # ----- 規格化 g2 プロット -----
    print("  [Step 7] Generating normalized g2 plot...")
    plot_normalized_g2(data_dict_matched, params, output_dir, fmt=plot_format, dpi=plot_dpi)
    print()

    # ----- Gamma vs q 解析 -----
    print("  [Step 8] Gamma vs q analysis...")
    q_arr = np.array([params[idx]['q'] for idx in matched_indices])
    gamma_arr = np.array([params[idx]['gamma'] for idx in matched_indices])
    gamma_err_arr = np.array([params[idx]['gamma_err'] for idx in matched_indices])

    fit_result = fit_gamma_vs_q(q_arr, gamma_arr, gamma_err_arr)

    print(f"    Fit result: Gamma = A * q^n")
    print(f"      A = {fit_result['A']:.6e} +/- {fit_result['A_err']:.6e}")
    print(f"      n = {fit_result['n']:.6f} +/- {fit_result['n_err']:.6f}")
    print(f"      R^2 = {fit_result['R2']:.6f}")
    print()

    # ----- Gamma vs q プロット -----
    print("  [Step 9] Generating Gamma vs q plot...")
    plot_gamma_vs_q(q_arr, gamma_arr, gamma_err_arr, fit_result,
                    output_dir, fmt=plot_format, dpi=plot_dpi)
    print()

    # ----- Gamma vs q データ保存 -----
    print("  [Step 10] Saving Gamma vs q data...")
    save_gamma_vs_q(output_dir, q_arr, gamma_arr, gamma_err_arr, fit_result)
    print()

    # ----- alpha vs q 解析・プロット -----
    print("  [Step 11] Generating alpha vs q plot...")
    alpha_arr = np.array([params[idx]['alpha'] for idx in matched_indices])
    alpha_err_arr = np.array([params[idx]['alpha_err'] for idx in matched_indices])

    plot_alpha_vs_q(q_arr, alpha_arr, alpha_err_arr,
                    output_dir, fmt=plot_format, dpi=plot_dpi)
    print()

    # ----- alpha vs q データ保存 -----
    print("  [Step 12] Saving alpha vs q data...")
    save_alpha_vs_q(output_dir, q_arr, alpha_arr, alpha_err_arr)
    print()

    # ----- フィッティング結果サマリー保存 -----
    print("  [Step 13] Saving fit summary...")
    save_fit_summary(output_dir, fit_result)
    print()

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """
    カレントディレクトリ以下を再帰探索し、StrExpPara_*.dat が存在する
    全ディレクトリを自動検出して順に処理する。
    出力は OUTPUT_ROOT 以下にサブフォルダ分けして保存する。
    """

    print("=" * 60)
    print("XPCS Data Processing Script (Batch Mode)")
    print("=" * 60)
    print()

    # ----- ディレクトリ探索 -----
    print(f"Searching for data directories under: {os.path.abspath(ROOT_DIR)}")
    print(f"  Pattern: {PARAM_PATTERN}")
    print()

    data_dirs = find_data_directories(ROOT_DIR, PARAM_PATTERN)

    if not data_dirs:
        print("ERROR: No directories containing parameter files found.")
        return

    print(f"Found {len(data_dirs)} data directory(ies):")
    for d in data_dirs:
        print(f"  {os.path.abspath(d)}")
    print()

    # ----- 各ディレクトリを順に処理 -----
    success_count = 0
    fail_count = 0

    for i, data_dir in enumerate(data_dirs, start=1):
        # 出力サブフォルダ名: ディレクトリの相対パスを使用
        rel_path = os.path.relpath(data_dir, ROOT_DIR)
        if rel_path == ".":
            subfolder_name = "root"
        else:
            # パス区切りをアンダースコアに変換してフラットなフォルダ名にする
            subfolder_name = rel_path.replace(os.sep, "_")

        output_dir = os.path.join(OUTPUT_ROOT, subfolder_name)

        print("=" * 60)
        print(f"[{i}/{len(data_dirs)}] Processing: {os.path.abspath(data_dir)}")
        print(f"  Output -> {os.path.abspath(output_dir)}")
        print("-" * 60)

        try:
            result = process_directory(data_dir, output_dir)
            if result:
                success_count += 1
                print(f"  Completed successfully.")
            else:
                fail_count += 1
                print(f"  Skipped (no valid data).")
        except Exception as e:
            fail_count += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        print()

    # ----- サマリー -----
    print("=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"  Total directories found:  {len(data_dirs)}")
    print(f"  Successfully processed:   {success_count}")
    print(f"  Failed / skipped:         {fail_count}")
    print(f"  Output root: {os.path.abspath(OUTPUT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()