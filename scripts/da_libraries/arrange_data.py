import glob
from collections import defaultdict

import pandas as pd
from scipy import signal
from tqdm.notebook import tqdm

from scripts.kubota_libraries.general_functions import *

from settings.general_variables import *


class DataOrganizer:
    """全データを整理するクラス"""

    def __init__(self, should_normalize_data: bool = False, should_flatten_data: bool = False,
                 m: int = 0, frequency_c: int = 0):
        """
        Args:
            should_normalize_data: データを平滑化するかどうかのブール値
            should_flatten_data: データを平滑化するかどうかのブール値
            m: 窓の大きさ
            frequency_c: カットオフ周波数
        """
        self.should_normalize_data = should_normalize_data

        self.arranged_strain_dataframe: pd.DataFrame = pd.DataFrame()
        self.split_strain_data_df: pd.DataFrame = pd.DataFrame()

        self.should_flatten_data = should_flatten_data
        self.data_already_flattened = False
        self.m = m
        self.frequency_c = frequency_c
        self.h_lw = None

    def _calculate_filter_coefficient(self):
        """
        フィルタの係数を計算する関数
        Returns: None
        """
        omega_c = 2.0 * np.pi * self.frequency_c / FREQUENCY_S
        a = 0.54

        h_l = [0 for _ in range(2 * self.m + 1)]
        for j in range(1, 2 * self.m + 1):
            if j == self.m + 1:
                continue

            h_l[j] = np.sin((j - self.m - 1) * omega_c) / (np.pi * (j - self.m - 1))

        h_l = np.array(h_l)

        w = [0 for _ in range(2 * self.m + 1)]
        for j in range(1, 2 * self.m + 1):
            w[j] = a + (1 - a) * np.cos((j - self.m - 1) * np.pi / self.m)

        w = np.array(w)

        h_lw = w * h_l

        h_l[self.m + 1] = omega_c / np.pi
        h_lw[self.m + 1] = h_l[self.m + 1]
        self.h_lw = np.reshape(h_lw, (h_lw.shape[0], 1))

    def calculate_filter_frequency_response(self, should_db: bool = True):
        """
        フィルタの周波数応答を計算し、その結果を返す関数。
        Args:
            should_db: デシベルに変換するかどうか
        Returns:
            w: 周波数レスポンス
            amp_dB: デシベル変換された振幅（should_db=Trueの場合）
        """
        w, H_lw = signal.freqz(self.h_lw)

        if should_db:
            amp_dB = 20 * np.log10(np.abs(H_lw) / np.max(np.abs(H_lw)))
        else:
            amp_dB = H_lw

        w = (w * FREQUENCY_S) / np.pi

        return w, amp_dB

    def _collect_arranged_strain_dataframes(self) -> None:
        """
        CSVデータを整理してデータフレームに変換し、self.arranged_strain_dataframesに追加します。
        Returns: None
        """

        strain_phases = {wind_environment: 0 for wind_environment in WIND_ENVIRONMENTS}
        self.arranged_strain_dataframe = pd.DataFrame(columns=["WE", "SP"] + DATA_COLUMN_NAMES)
        for original_data_path in ORIGINAL_DATA_PATHS:
            if NO_FLAPPING_DATA_FILE_NAME:
                print("Offset is conducted")
                strain_dataframe_no_flapping = load_csv_and_transform_to_dataframe(
                    rf"{original_data_path}\{NO_FLAPPING_DATA_FILE_NAME}.CSV"
                )
                strain_mean_series_no_flapping = strain_dataframe_no_flapping.mean()
            else:
                print("No offset is conducted")
                strain_mean_series_no_flapping = 0

            for wind_environment in tqdm(WIND_ENVIRONMENTS):
                flapping_folder_path = rf"{original_data_path}\{FLAPPING_DATA_FOLDER_NAME}\{wind_environment}"
                flapping_file_paths: list = glob.glob(rf"{flapping_folder_path}\*.CSV")

                print(wind_environment, end=": ")
                print(len(flapping_file_paths))

                for flapping_file_path in flapping_file_paths:
                    strain_phases[wind_environment] += 1
                    strain_df: pd.DataFrame = load_csv_and_transform_to_dataframe(flapping_file_path)
                    strain_df -= strain_mean_series_no_flapping

                    if self.should_normalize_data:
                        data_scaled = normalize_dataframe(strain_df)
                        strain_df = pd.DataFrame(data_scaled, columns=DATA_COLUMN_NAMES)

                    strain_df = cancel_offset_of_encoder(strain_df)

                    strain_df["WE"] = wind_environment
                    strain_df["SP"] = int(strain_phases[wind_environment])

                    self.arranged_strain_dataframe = pd.concat(
                        [self.arranged_strain_dataframe, strain_df], ignore_index=True)
        self._reduce_memory("whole")
        return

    def _count_data_per_wind_environment(self, whole_or_divided, wind_environment):
        """
        データを平滑化する関数
        Args:
            whole_or_divided: 分割前か後か
            wind_environment: どの風環境か
        Returns:
            w: 周波数レスポンス
            amp_dB: デシベル変換された振幅（should_db=Trueの場合）        """
        if whole_or_divided == "whole":
            df = self.arranged_strain_dataframe
        elif whole_or_divided == "divided":
            df = self.split_strain_data_df
        else:
            return 0

        # 'WE'がwind_environmentの値と一致する行をフィルタリング
        filtered_df = df[df['WE'] == wind_environment]

        # フィルタリングされたデータフレームの 'SP' 列の一意な値の数を取得
        unique_values_count = filtered_df['SP'].nunique()
        return unique_values_count

    def _flatten_data(self):
        """
        データを平滑化する関数
        Returns: None
        """
        if self.data_already_flattened:
            print("Data is already flattened.")
            return

        smoothed_strain_df = pd.DataFrame(columns=["WE", "SP"] + DATA_COLUMN_NAMES)

        for wind_environment in tqdm(WIND_ENVIRONMENTS):
            dataframes_count = self._count_data_per_wind_environment("whole", wind_environment)
            strain_df = self.arranged_strain_dataframe.loc[self.arranged_strain_dataframe["WE"] == wind_environment]
            for i in range(dataframes_count):
                tmp_strain_data_per_strain_phase = strain_df.loc[strain_df["SP"] == i + 1].copy()
                tmp_strain_data_per_strain_phase = tmp_strain_data_per_strain_phase.loc[:, DATA_COLUMN_NAMES]
                self.arranged_strain_dataframe.drop(tmp_strain_data_per_strain_phase.index)

                tmp_strain_data_per_strain_phase_np = tmp_strain_data_per_strain_phase.to_numpy()
                tmp_smoothed_strain_np = signal.convolve2d(tmp_strain_data_per_strain_phase_np, self.h_lw, "valid")
                tmp_smoothed_strain_df = pd.DataFrame(tmp_smoothed_strain_np, columns=DATA_COLUMN_NAMES)
                tmp_smoothed_strain_df["WE"] = wind_environment
                tmp_smoothed_strain_df["SP"] = i + 1

                smoothed_strain_df = pd.concat([smoothed_strain_df, tmp_smoothed_strain_df], ignore_index=True)


        self.arranged_strain_dataframe = smoothed_strain_df

        self.data_already_flattened = True

    def _reduce_memory(self, whole_divided):
        if whole_divided == "whole":
            df = self.arranged_strain_dataframe
        elif whole_divided == "divided":
            df = self.split_strain_data_df
        else:
            return

        df["WE"] = df["WE"].astype(str)
        # df["SP"] = df["SP"].astype(np.int8)

        col = "SP"
        c_min = df[col].min()
        c_max = df[col].max()
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            df[col] = df[col].astype(np.int64)

        return

    def update_filter_parameters(self, m, frequency_c) -> None:
        """
        平滑化のパラメータmとfrequency_cを更新し、フィルター係数を再計算する関数。
        Args:
            m: 平滑化のパラメータm (int)
            frequency_c: 平滑化のパラメータfrequency_c (int)
        """

        self.m = m
        self.frequency_c = frequency_c

        self._calculate_filter_coefficient()

        return


class StrainDataSplitter(DataOrganizer):
    """
    ひずみデータの管理、CSVとデータフレームの相互変換、周期の抽出を行うクラス
    """

    def __init__(self, should_normalize_data: bool = False, should_flatten_data: bool = False,
                 m: int = 0, frequency_c: int = 0):
        """
        Args:
            should_normalize_data: データを標準化するかどうか (bool)
            should_flatten_data: データを平滑化するかどうか (bool)
            m: 平滑化のパラメータm (int)
            frequency_c: 平滑化のパラメータfrequency_c (int)
        """

        super().__init__(should_normalize_data, should_flatten_data, m, frequency_c)
        self._collect_arranged_strain_dataframes()

        self.strain_period = None

        self.split_num = None
        self.time_over_period = None
        self.should_split_data_per_period = False
        self.df_size = None

        if self.should_flatten_data:
            if self.m == 0 or self.frequency_c == 0:
                raise ValueError("Please define m and frequency_c")
            self._calculate_filter_coefficient()

        self.strain_period: int

        print(f"Found {len(WIND_ENVIRONMENTS)} folders.")
        print("After checking the graph of raw data (arranged_strain_dataframes), implement get_split_strain_data_dfs.")

    def __calculate_split_indices(self, mode: str):
        """
        分割するためのインデックスを取得する関数。
        Args:
            mode: 分割方法。'random'または'period'が指定可能。
        Returns:
            indices: データを手動で分割するためのインデックス (numpy array)
        """

        raise EOFError("この関数の使用は出来ません。")

        # if not self.df_size:
        #     for i, flapping_file_path in enumerate(self.arranged_strain_dataframes):
        #         if i >= 1:
        #             break
        #         self.df_size = len(self.arranged_strain_dataframes[flapping_file_path][0])
        #
        # if self.should_flatten_data:
        #     self.__flatten_data()
        #
        # current_df_size = self.df_size  # Set default value for current_df_size
        #
        # for i, flapping_file_path in enumerate(self.arranged_strain_dataframes):
        #     if i >= 1:
        #         break
        #     current_df_size = len(self.arranged_strain_dataframes[flapping_file_path][0])
        #
        # if mode == 'random':
        #     print(DEPRECIATING_WARNING)
        #     indices = np.arange(0, current_df_size, self.df_size // self.split_num - 1)
        # elif mode == 'period':
        #     indices = np.arange(0, current_df_size,
        #                         int(np.round((current_df_size / TIME_S / FLAPPING_FREQUENCY) * self.time_over_period)))
        # else:
        #     raise ValueError(f'Invalid mode: {mode}. Must be "random" or "period".')
        #
        # return indices

    def __create_data_save_directories(self, root_data_location):
        """
        整列したデータを保存するためのフォルダを作成する関数。
        Args:
            root_data_location: ルートデータ位置 (string)
        Returns:
            parent_folder_name: 作成された親フォルダの名前 (string)
        """

        if self.should_flatten_data:
            root_data_location += rf"\{FLATTEN_DATA_FOLDER_NAME}"
        else:
            root_data_location += rf"\{RAW_DATA_FOLDER_NAME}"

        create_directory(rf"{root_data_location}")

        if self.split_num is None:
            if self.time_over_period is None:
                raise ValueError("time_over_periodを定義してください。")
                # parent_folder_name = rf"{root_data_location}\arranged_data"
            else:
                parent_folder_name = rf"{root_data_location}\data_split_by_{self.time_over_period}_period"
        else:
            parent_folder_name = rf"{root_data_location}\data_split_to_{self.split_num}"

        create_directory(parent_folder_name)

        if self.should_flatten_data:
            parent_folder_name = rf"{parent_folder_name}\cutoff{self.frequency_c}_m{self.m}"
            create_directory(parent_folder_name)

        if self.should_normalize_data:
            parent_folder_name = rf"{parent_folder_name}\{NORMALIZED_DATA_FOLDER_NAME}"
        else:
            parent_folder_name = rf"{parent_folder_name}\{NOT_NORMALIZED_DATA_FOLDER_NAME}"
        create_directory(parent_folder_name)

        return parent_folder_name

    def save_data(self, file_format: str) -> None:
        """
        この関数は指定されたフォーマットでデータを保存します。サポートされているフォーマットは 'csv', 'pkl', 'npy' です。
        'csv' および 'pkl' の場合、split_strain_data_dfs の各データフレームは別々のファイルに保存されます。
        'npy' の場合は、すべてのデータフレームが一つの numpy 配列に統合され、その配列がファイルに保存されます。

        Args:
            file_format: データの保存形式。'csv'、'pkl'、または 'npy'。

        Raises:
            ValueError: サポートされていないフォーマットが指定された場合。
        """
        ...

        saving_methods = {
            'csv': lambda d, path: d.to_csv(path, index=False),
            'pkl': lambda d, path: d.to_pickle(path),
            'npy': lambda d, path: np.save(path, d)
        }

        if file_format not in saving_methods:
            raise ValueError(f"Unsupported format: {file_format}. Choose from 'csv', 'pkl', 'npy'.")

        if file_format != 'npy':
            parent_folder_name = self.__create_data_save_directories(SPLIT_DATA_PATH)
            saving_methods[file_format](self.split_strain_data_df,
                                        rf"{parent_folder_name}\organized_data.{file_format}")
        else:
            input_datasets = []
            answer_datasets = []
            skip_count = 0

            for i, wind_environment in enumerate(WIND_ENVIRONMENTS):
                unique_values_count = self._count_data_per_wind_environment("divided", wind_environment)
                answer_datasets += [i for _ in range(unique_values_count)]
                data_df = self.split_strain_data_df.loc[self.split_strain_data_df["WE"] == wind_environment].copy()
                for strain_phase in range(1, 1 + unique_values_count):
                    tmp_data_df = data_df.loc[data_df["SP"] == strain_phase]
                    tmp_data_df = tmp_data_df[DATA_COLUMN_NAMES].copy()
                    if i == 0 and strain_phase == 1:
                        input_datasets = [tmp_data_df.values]
                        continue
                    if len(input_datasets[0]) == tmp_data_df.shape[0]:
                        input_datasets += [tmp_data_df.values]
                    else:
                        answer_datasets = answer_datasets[:-1]
                        skip_count += 1
                        continue
            input_datasets = np.array(input_datasets)
            answer_datasets = np.array(answer_datasets)

            parent_folder_name = self.__create_data_save_directories(TRAINING_FILE_PATH)
            print(f"Didn't save {skip_count} dataframes.")

            saving_methods[file_format](input_datasets, rf"{parent_folder_name}\input_datasets.npy")
            saving_methods[file_format](answer_datasets, rf"{parent_folder_name}\answer_datasets.npy")

    def split_strain_data(self, mode: str, value: float) -> None:
        """
        データを分割する関数。
        Args:
            mode: 分割方法。'random'または'period'が指定可能。
            value: "random"モードの場合は分割数split_num、"period"の場合はtime_over_periodを代入
            ※time_over_periodは周期に対する無次元時間（例：0.5に設定したら周期の半分でカット
        """

        if mode == 'random':
            print(DEPRECIATING_WARNING)
            # self.should_split_data_per_period = False
            # self.split_num = value
            # self.time_over_period = None
            # indices = self.__calculate_split_indices(mode)
        elif mode == 'period':
            self.should_split_data_per_period = True
            self.split_num = None
            self.time_over_period = value
        else:
            raise ValueError(f'Invalid mode: {mode}. Must be "random" or "period".')

        if self.should_flatten_data:
            self._flatten_data()

        # データの初期化
        self.split_strain_data_df = pd.DataFrame(columns=["WE", "SP"] + DATA_COLUMN_NAMES)

        for wind_environment in WIND_ENVIRONMENTS:
            strain_df = self.arranged_strain_dataframe.loc[
                self.arranged_strain_dataframe["WE"] == wind_environment].copy()

            strain_df["SP"] = strain_df["SP"].astype(np.int64)

            num_files: int = int(self._count_data_per_wind_environment("whole", wind_environment))
            split_phase = 0
            for string_phase in range(1, num_files + 1):

                strain_df_per_split_phase = strain_df.loc[strain_df["SP"] == string_phase].copy()
                strain_df_per_split_phase.reset_index(drop=True, inplace=True)

                num_row_per_split_phase = int((FREQUENCY_S / FLAPPING_FREQUENCY) * self.time_over_period)

                # 'SP' 列の作成
                num_rows = strain_df_per_split_phase.shape[0]  # データフレームの行数を取得
                num_splits = num_rows // num_row_per_split_phase  # データを分割するセクション数を決定

                # 各セクションに対応する値を設定
                for i in range(num_splits):
                    split_phase += 1
                    strain_df_per_split_phase.loc[i * num_row_per_split_phase: (i + 1) * num_row_per_split_phase - 1,
                    'SP'] = split_phase

                # 行が num_row_per_split_phase で割り切れない場合、余り部分の行を削除
                if num_rows % num_row_per_split_phase != 0:
                    strain_df_per_split_phase = strain_df_per_split_phase.iloc[:num_splits * num_row_per_split_phase]
                self.split_strain_data_df = pd.concat([self.split_strain_data_df, strain_df_per_split_phase],
                                                      ignore_index=True)

            print(wind_environment, end=": ")
            print(self._count_data_per_wind_environment("divided", wind_environment))
        self._reduce_memory("divided")
        return
