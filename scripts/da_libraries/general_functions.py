import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from settings.general_variables import *


def cancel_offset_of_encoder(df):
    for column in df.columns:
        if column[:2] == 'EN':
            min_value = df[column].min()
            df[column] = df[column] - min_value
    return df


def convert_csv_to_dataframe(start: int, end: int, file_path: str) -> pd.DataFrame:
    """
    CSVデータを整理されたデータフレームに変換する関数

    Args:
        start (int): CSVデータからの開始行
        end (int): CSVデータからの終了行
        file_path (str): CSVデータファイルのパス

    Returns:
        df (DataFrame): 整理されたデータフレーム
    """
    csv_file = pd.read_csv(file_path, names=["1", "SG1", "SG2", "SG3", "2"])
    df = csv_file.iloc[start: end, 1:4]

    for i in range(3):
        df[f"SG{i + 1}"] = pd.to_numeric(df[f"SG{i + 1}"], downcast="float")

    return df


def create_directory(directory_path) -> None:
    """
    指定されたパスにディレクトリを作成します。既に存在する場合は何もしません
    Args:
        directory_path: 作成するディレクトリのパス
    Returns:
        None
    """
    try:
        os.makedirs(directory_path)
    except FileExistsError:
        pass


def format_number(number):
    prefixes = {
        -15: 'f',  # femto
        -12: 'p',  # pico
        -9: 'n',  # nano
        -6: 'μ',  # micro
        -3: 'm',  # milli
        0: '',  # no prefix
        3: 'k',  # kilo
        6: 'M',  # Mega
        9: 'G',  # Giga
        12: 'T',  # Tera
        15: 'P',  # Peta
        18: 'E',  # Exa
    }

    if number == 0:
        return 0, ''

    exponent = min(max(int(math.log10(abs(number)) // 3 * 3), -15), 18)
    value = number / (10 ** exponent)
    return value, prefixes[exponent]


def load_csv_and_transform_to_dataframe(file_path: str, header=14) -> pd.DataFrame:
    """
    CSVファイルを読み込み、特定の範囲のデータを選択してデータフレームに変換します。
    header引数でヘッダーの有無を選択できます。

    Args:
        file_path: 読み込むCSVファイルのパス
        header: CSVファイルのヘッダー有無
    Returns:
        pandas DataFrame
    """
    selected_data = None
    if DATA_LOGGER_TYPE == "DL850E":
        csv_file = pd.read_csv(file_path, header=header)
        csv_file.reset_index(drop=True, inplace=True)
        selected_data = csv_file.iloc[:, :len(DATA_COLUMN_NAMES)]
        selected_data.columns = DATA_COLUMN_NAMES
        for data_column_name in DATA_COLUMN_NAMES:
            selected_data = selected_data.copy()
            selected_data.loc[:, data_column_name] = pd.to_numeric(selected_data[data_column_name], downcast="float")

    elif DATA_LOGGER_TYPE == "EDX-10A":
        csv_file = pd.read_csv(file_path, skiprows=15, encoding='Shift_JIS')
        selected_data = csv_file.iloc[:, 1:]
        selected_data.columns = DATA_COLUMN_NAMES
        for data_column_name in DATA_COLUMN_NAMES:
            selected_data = selected_data.copy()
            selected_data.loc[:, data_column_name] = pd.to_numeric(selected_data[data_column_name], downcast="float")

    return selected_data


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    データのスケーリング/標準化

    Args:
        dataframe: データフレーム
    Returns:
        data_scaled: 標準化されたデータ
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataframe)
    return data_scaled


def perform_pca_on_dataframe(normalized_dataframe: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    データのpcaを実行

    Args:
        normalized_dataframe: 正規化されたデータフレーム
        n_components: 何次元に削減するか
    Returns:
        data_pca: pcaされたデータ
    """

    # PCAのインスタンスを作成
    pca = PCA(n_components=n_components).fit(normalized_dataframe)

    # 固有値とその出力
    eigenvalues = pca.explained_variance_
    print('Eigenvalues:', eigenvalues)

    # 寄与率とその出力
    explained_variance_ratio = pca.explained_variance_ratio_
    print('Explained Variance Ratio:', explained_variance_ratio)

    # 累積寄与率とその出力
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print('Cumulative Explained Variance:', cumulative_explained_variance)

    # 主成分負荷量

    # 主成分負荷量をデータフレームとして表示
    loadings = pca.components_
    df_loadings = pd.DataFrame(loadings, columns=DATA_COLUMN_NAMES,
                               index=[f'PC{i + 1}' for i in range(pca.n_components_)])
    print('Loadings in DataFrame:\n', df_loadings)

    # PCAのフィットと変換
    data_pca = pca.fit_transform(normalized_dataframe)

    return data_pca


def split_data(df: pd.DataFrame, indices: np.array) -> list:
    """
    与えられたインデックスに基づいてデータフレームを分割する関数

    Args:
        df: 分割するデータフレーム
        indices: 分割する位置を示すインデックスのリスト

    Returns:
        list: 分割されたデータフレームのリスト
    """
    split_dataframes = []
    for i in range(len(indices) - 1):
        split_dataframes.append(df.iloc[indices[i]: indices[i + 1], :])
    return split_dataframes


def substitute_zero_with_input(x: int, q: str):
    """
    xが0の場合、質問qに基づく入力を返し、それ以外の場合はxをそのまま返す関数

    Args:
        x (int): 判断する整数
        q (str): 入力要求する質問

    Returns:
        x (int) or user input (str): xが0でない場合はx、0の場合はユーザーからの入力
    """
    if x == 0:
        return input(q)
    else:
        return x
