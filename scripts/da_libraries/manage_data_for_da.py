from glob import glob
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

from scripts.kubota_libraries.general_functions import *
from scripts.kubota_libraries.arrange_data import DataOrganizer

from settings.general_variables import *


class StrainDataManager(DataOrganizer):
    """
    ひずみデータの個別参照、統計解析を行うクラス
    """

    def __init__(self):
        """
        コンストラクタ
        """

        super().__init__()
        self.whole_or_divided = None
        self.selected_folder_path = self._select_data_folder_for_data_analytics()

        if self.whole_or_divided == "whole":
            self._collect_arranged_strain_dataframes()
        elif self.whole_or_divided == "divided":
            self.__get_split_strain_dataframes()

        print(IMPLEMENTATION_ORDER_WARNING)

        self.selected_wind_environment: str = ""
        self.selected_data_index: int = -1
        self.num_split_phase: int = 0

    def change_data(self):
        """
        同じフォルダー（風向・風速が同じデータ群）の中でデータを変更する関数
        Returns:
            None
        """

        if self.whole_or_divided == "divided":
            self.selected_data_index = int(
                input(f"{QUESTION_PROMPT_TARGET_DATA_INDEX}\n1 - {self.num_split_phase}"))
            print(f"# data: {self.selected_data_index}")
        elif self.whole_or_divided == "whole":
            self.selected_data_index = 0
            print(f"# data: All")
        return

    def choose_data(self):
        """
        フォルダー（風向・風速が同じデータ群）およびデータを選択する関数
        Returns:
            None
        """
        flapping_folder_explanation = ""
        for i, wind_environment in enumerate(WIND_ENVIRONMENTS):
            flapping_folder_explanation += f"\n{i}: {wind_environment}"
        chosen_number = int(input(f"{QUESTION_PROMPT_WIND_INFO}{flapping_folder_explanation}"))
        print(f"Folder name: {WIND_ENVIRONMENTS[chosen_number]}")
        self.selected_wind_environment = WIND_ENVIRONMENTS[chosen_number]
        if self.whole_or_divided == "whole":
            df = self.arranged_strain_dataframe
        elif self.whole_or_divided == "divided":
            df = self.split_strain_data_df
        else:
            return
        self.num_split_phase = \
            int(df.loc[df["WE"] == self.selected_wind_environment]["SP"].max())
        self.change_data()
        return

    def __get_split_strain_dataframes(self) -> None:
        """
        外部の”csv_to_arranged_df”, ”csv_to_arranged_df2”を使って，綺麗なdataframesv_to_arranged_dfに変換して
        self.arranged_strain_dataframesに入れる関数
        Returns:
            None
        """

        flapping_folder_paths = sorted(glob(f"{self.selected_folder_path}\\*"))
        print(flapping_folder_paths)
        for flapping_folder_path in tqdm(flapping_folder_paths):
            self.split_strain_data_df = pd.read_pickle(flapping_folder_path)

        return

    def __get_whole_strain_dfs(self) -> None:
        """
        外部の”csv_to_arranged_df”, ”csv_to_arranged_df2”を使って，綺麗なdataframesv_to_arranged_dfに変換して
        self.arranged_strain_dataframesに入れる関数
        Returns:
            None
        """
        if NO_FLAPPING_DATA_FILE_NAME:
            neither_flapping_nor_wind_df = load_csv_and_transform_to_dataframe(
                rf"{self.selected_folder_path}\{NO_FLAPPING_DATA_FILE_NAME}.CSV"
            )
            neither_flapping_nor_wind_mean_series = neither_flapping_nor_wind_df.mean()
        else:
            neither_flapping_nor_wind_mean_series = 0

        for wind_environment in tqdm(WIND_ENVIRONMENTS):
            flapping_file_paths: list = glob(rf"{wind_environment}\*.CSV")

            folder_name: str = wind_environment.split("\\")[-1]

            for flapping_file_path in flapping_file_paths:
                strain_df: pd.DataFrame = load_csv_and_transform_to_dataframe(flapping_file_path)
                strain_df -= neither_flapping_nor_wind_mean_series

                self.split_strain_data_df[folder_name].append(strain_df)
        return

    def plot_data(self, title: str = None, figure_size: tuple = (20, 15), linewidth=1,
                  tick_font_size: int = 10, label_font_size: int = 40, title_fontsize=50,
                  y_lim=None, font_name="Times New Roman",
                  folder_name: str = "front_horizontal_1.5mps_with_flapping") -> plt.figure:
        """
        保持しているデータをプロットする関数．
        Args:
            title: タイトル（デフォルトはNone）
            figure_size: フィギュアサイズ（デフォルトは(20, 15)）
            linewidth: 線の太さ（デフォルトは1）
            tick_font_size: ティックフォントサイズ（デフォルトは10）
            label_font_size: ラベルフォントサイズ（デフォルトは40）
            title_fontsize: タイトルフォントサイズ（デフォルトは50）
            y_lim: Y軸の範囲（デフォルトはNone）
            font_name: フォント名（デフォルトは"Times New Roman"）
            folder_name: フォルダ名（デフォルトは"front_horizontal_1.5mps_with_flapping"）
        Returns:
            plt.figure: 描画したmatplotlibのfigureオブジェクト
        """

        self.__raise_data_not_selected_error()

        if self.whole_or_divided == "divided":
            pass
        else:
            return None

        if title is None:
            title = f"{self.selected_wind_environment} #{self.selected_data_index}"
        else:
            title = title
        df_copy = self.show_data()

        dx = 1 / FREQUENCY_S
        time_ticks = np.array([t * dx for t in range(len(df_copy))])
        df_copy = df_copy.set_index(pd.Index(time_ticks))

        sg_cols = [col for col in df_copy.columns if col.startswith('SG')]
        en_cols = [col for col in df_copy.columns if col.startswith('EN')]

        fig, axs = plt.subplots(2, 1, figsize=figure_size, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        df_copy[sg_cols].plot.line(ax=axs[0],
                                   color=[sg_color for sg_color in SG_COLORS],
                                   linewidth=linewidth,
                                   grid=True,
                                   fontsize=30,
                                   mark_right=False,
                                   xlim=(0, len(df_copy) * dx)
                                   )

        if en_cols:
            df_copy[en_cols].plot.line(ax=axs[1],
                                       color=[en_color for en_color in EN_COLORS],
                                       linewidth=linewidth,
                                       grid=True,
                                       fontsize=30,
                                       mark_right=False,
                                       xlim=(0, len(df_copy) * dx)
                                       )
            axs[1].set_ylim((-0.1, 3.5))
            axs[1].set_ylabel("volt", fontdict={"fontsize": label_font_size, 'fontname': font_name})

        if y_lim:
            axs[0].set_ylim(y_lim)

        for ax in axs:
            ax.set_xlabel("Time: t (s)", fontdict={"fontsize": label_font_size, 'fontname': font_name})
            ax.tick_params(labelsize=tick_font_size)

        axs[0].set_ylabel("Strain: λ (μSTR)", fontdict={"fontsize": label_font_size, 'fontname': font_name})
        plt.suptitle(title, fontsize=title_fontsize, fontname=font_name)

        for ax in axs:
            ax.legend(prop={'size': label_font_size})

        return fig

    def __raise_data_not_selected_error(self) -> None:
        """
        選択されていないデータに対してエラーを発生させる関数
        Returns:
            None（ただし条件が満たされていない場合はValueErrorを発生）
        """
        if self.selected_wind_environment == "":
            raise ValueError("choose_data()を実行して、フォルダーを選択してください")

        if self.selected_data_index == -1:
            raise ValueError("change_data()を実行して、データ番号を選択してください")

        return

    def _select_data_folder_for_data_analytics(self):
        """
        データ分析のために、選択したフォルダを返す関数。

        Returns:
            chose_folder (str): 選択したフォルダのパス
        """

        def user_choice(question, choices):
            while True:
                options = "\n".join([f"{i + 1}: {c}" for i, c in enumerate(choices)])
                user_input = input(f"{question}\n{options}\n\n{QUESTION_PROMPT_END}\n")
                if user_input.lower() == "end":
                    print(NO_FILE_CHOSEN_WARNING)
                    return None
                if 0 < int(user_input) <= len(choices):
                    print(choices[int(user_input) - 1] + "を選択")
                    return choices[int(user_input) - 1]

        choice = input(QUESTION_PROMPT_WHOLE_OR_SPLIT)
        if choice not in ["1", "2"]:
            print("1, 2で選択してください")
            return None

        self.whole_or_divided = "whole" if choice == "1" else "divided"
        print("Whole or Split: ", "Whole" if choice == "1" else "Split")

        base_folder = ORIGINAL_DATA_PATHS[0] if choice == "1" else SPLIT_DATA_PATH
        if choice == "1":
            return base_folder

        data_type_choice = input(QUESTION_PROMPT_FLATTEN_OR_RAW)
        chosen_folder_path = os.path.join(
            base_folder, FLATTEN_DATA_FOLDER_NAME if data_type_choice == "1" else RAW_DATA_FOLDER_NAME)
        print("Flatten or Raw: ", "Flatten" if data_type_choice == "1" else "Raw")

        data_choices_split_num = [os.path.basename(path) for path in sorted(glob(f"{chosen_folder_path}\\*"))]
        chosen_split_num = user_choice(QUESTION_PROMPT_SPLIT_NUM, data_choices_split_num)
        if chosen_split_num is None:
            return None

        chosen_folder_path = os.path.join(chosen_folder_path, chosen_split_num)
        if data_type_choice == "1":
            data_choices_flatten_parameter = [os.path.basename(path) for path in
                                              sorted(glob(f"{chosen_folder_path}\\*"))]
            chosen_flatten_parameter = user_choice(QUESTION_PROMPT_FLATTENING_PARAMETER, data_choices_flatten_parameter)
            if chosen_flatten_parameter is None:
                return None
            chosen_folder_path = os.path.join(chosen_folder_path, chosen_flatten_parameter)

        data_normalization_choice = input(QUESTION_PROMPT_NORMALIZE_DATA)
        chosen_folder_path = os.path.join(
            chosen_folder_path,
            NORMALIZED_DATA_FOLDER_NAME if data_normalization_choice.upper() == "Y" else NOT_NORMALIZED_DATA_FOLDER_NAME)
        print("NORMALIZED: ", "YES" if data_normalization_choice.upper() == "Y" else "NO")

        return chosen_folder_path

    def show_data(self) -> pd.DataFrame:
        """
        現在選択されているデータを表示する関数
        Returns:
            pandas.DataFrame: 現在選択されているデータ
        """
        self.__raise_data_not_selected_error()

        if self.whole_or_divided == "whole":
            chosen_strain_data = self.arranged_strain_dataframe.loc[
                self.arranged_strain_dataframe["WE"] == self.selected_wind_environment].copy()
        elif self.whole_or_divided == "divided":
            chosen_strain_data = self.split_strain_data_df.loc[
                self.split_strain_data_df["WE"] == self.selected_wind_environment].copy()
        else:
            return pd.DataFrame()

        if self.selected_data_index > 0:
            chosen_strain_data = chosen_strain_data.loc[chosen_strain_data["SP"] == self.selected_data_index]

        chosen_strain_data.reset_index(drop=True, inplace=True)

        return chosen_strain_data

    """
    --------------------------------------------------------------------------------------------------------------------
    上は個別参照。
    下は統計解析。
    --------------------------------------------------------------------------------------------------------------------
    """

    def compute_average(self, should_normalize: bool):
        """
        各データフレームの平均値を計算し、それらを連結した新しいデータフレームを作成します。

        Returns:
            pd.DataFrame: 平均値を含む新しいデータフレーム
        """
        grouped = self.split_strain_data_df.groupby(['WE', 'SP'])
        averages_df = grouped.mean().reset_index()

        if should_normalize:
            columns_to_normalize = averages_df.columns.difference(['WE', 'SP'])
            averages_df[columns_to_normalize] = normalize_dataframe(averages_df[columns_to_normalize])

        return averages_df

    def compute_pca(self, n_components: int):
        """
        PCAを実行

        Aggs:
            n_components: 何次元にするか

        Returns:
            pd.DataFrame: PCAをした新しい値
        """

        data = self.compute_average(should_normalize=True)

        df_pca_subset = data[DATA_COLUMN_NAMES]

        pca_result = perform_pca_on_dataframe(df_pca_subset, n_components)
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i + 1}' for i in range(n_components)])
        pca_df = pd.concat([data[['SP', 'WE']].reset_index(drop=True), pca_df], axis=1)

        return pca_df

    @staticmethod
    def plot_combinations_2d(df, data_columns, nrows=2, ncols=4):
        """
        Plot sets of 2D combinations of data columns in separate windows, each containing 8 graphs, colored by category.

        Args:
        - df: DataFrame containing the data.
        - data_columns: List of columns to plot.
        - nrows: Number of rows in each subplot grid.
        - ncols: Number of columns in each subplot grid.
        Returns:
        - figs: List of matplotlib figure objects.
        """
        label_combinations = list(combinations(data_columns, 2))

        num_plots_per_window = nrows * ncols

        figs = []

        for i in range(0, len(label_combinations), num_plots_per_window):
            num_subplots = min(num_plots_per_window, len(label_combinations) - i)
            num_rows = (num_subplots + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=(16, 24), squeeze=False)

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

            window_combinations = label_combinations[i:i + num_plots_per_window]

            axes_flat = axes.flatten()

            for j, comb in enumerate(window_combinations):
                ax = axes_flat[j]
                for k, category in enumerate(df["WE"].unique()):
                    subset = df[df["WE"] == category]
                    ax.scatter(subset[comb[0]], subset[comb[1]], label=f"WE={category}",
                               c=(CLASSES_COLORS + EN_COLORS)[k])

                ax.set_xlabel(comb[0])
                ax.set_ylabel(comb[1])
                ax.set_title(f'{comb[0]} vs {comb[1]}')
                ax.legend()

            for l in range(num_subplots, num_plots_per_window):
                print(l)
                axes.flatten()[l].set_visible(False)

            plt.tight_layout()
            plt.show()

            figs.append(fig)

        return figs

    @staticmethod
    def plot_combinations_3d(df, data_columns):
        """
        Plot every combination of data columns in 3D, colored by category.

        Args:
        - df: DataFrame containing the data.
        - data_columns: List of columns to plot.
        """
        # Create combinations of all data columns for 3D plots
        label_combinations = list(combinations(data_columns, 3))

        # Plot each combination
        for comb in label_combinations:
            fig = plt.figure(figsize=(16, 24))

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

            ax = fig.add_subplot(111, projection='3d')
            for i, category in enumerate(df["WE"].unique()):
                subset = df[df["WE"] == category]
                ax.scatter(subset[comb[0]], subset[comb[1]], subset[comb[2]], label=f"{'WE'}={category}",
                           c=(CLASSES_COLORS + EN_COLORS)[i])

            ax.set_xlabel(comb[0])
            ax.set_ylabel(comb[1])
            ax.set_zlabel(comb[2])
            ax.set_title(f'{comb[0]}, {comb[1]}, {comb[2]}')
            ax.legend()
            plt.show()
        return fig

    def display_assemble_average_plot(self, en_y_lim=(), sn_y_lim=()):

        if self.whole_or_divided == "whole":
            pass
        else:
            print("wholeの時のみ実行可能です")
            return

        encoder_column = ""
        for c in DATA_COLUMN_NAMES:
            if "EN" in c:
                encoder_column = c
                break
        if encoder_column == "":
            print("Encoder利用時のみ実行可能です")
            return

        df = self.show_data().drop(['WE', 'SP'], axis=1)

        # Convert 'EN1' column to categories ranging from 0 to ENCODER_RESOLUTION-1
        df[encoder_column] = pd.cut(df[encoder_column], bins=ENCODER_RESOLUTION, labels=False)

        # Drop rows from 'EN1' becomes ENCODER_RESOLUTION-1 for the last time
        last_idx = df[df[encoder_column] == ENCODER_RESOLUTION - 1].index[-1]
        df = df.iloc[:last_idx]

        # Drop rows until 'EN1' becomes 0 for the first (or second) time
        first_zero_idx = df[df[encoder_column] == 0].index[0]
        second_zero_idx = df[df[encoder_column] == 0].index[1] if df.iloc[0][encoder_column] == 0 else first_zero_idx
        df = df.iloc[second_zero_idx + 1:]

        df.reset_index(drop=True, inplace=True)

        flapping_period_block_size = round(FREQUENCY_S / FLAPPING_FREQUENCY)

        # 周期ごとにFP列の値を決定

        ## "EN1"が0になる点（周期が開始する点）をリストに保存
        flapping_period_start_indexes_series = df[df[encoder_column] == 0].index.to_series().diff().gt(
            flapping_period_block_size / ENCODER_RESOLUTION)
        flapping_period_start_indexes_series = flapping_period_start_indexes_series[
            flapping_period_start_indexes_series == True].index
        flapping_period_start_indexes = [0] + list(flapping_period_start_indexes_series)

        for i in range(len(flapping_period_start_indexes) - 1):
            df.loc[flapping_period_start_indexes[i]: flapping_period_start_indexes[i + 1], 'FP'] = i

        # Group by 'FP' and calculate the size of each group
        group_sizes = df.groupby('FP').size()

        # Calculate the mean and standard deviation of the group sizes
        mean_size = group_sizes.mean()
        std_size = group_sizes.std()

        # Identify the outliers
        lower_bound = mean_size - std_size
        upper_bound = mean_size + std_size
        outliers = group_sizes[(group_sizes < lower_bound) | (group_sizes > upper_bound)]

        print(f"{outliers.shape[0]}周期分削除します")

        # Remove rows belonging to outlier groups
        df = df[~df['FP'].isin(outliers.index)]

        # Group by 'FP' and calculate the size of each group again
        group_sizes_cleaned = df.groupby('FP').size()

        # Find the size of the smallest group
        min_group_size = group_sizes_cleaned.min()

        # Trim each group to the size of the smallest group
        df = df.groupby('FP').apply(lambda x: x.iloc[:min_group_size]).reset_index(drop=True)

        # Create a dictionary mapping the old 'FP' values to new values
        fp_mapping = {old_fp: new_fp for new_fp, old_fp in enumerate(df['FP'].unique())}

        # Apply the mapping to the 'FP' column
        df['FP'] = df['FP'].map(fp_mapping)

        dt = 1 / min_group_size

        # 出力の時間を表す行ベクトル
        t_interp = np.arange(0, 1 + dt, dt)

        # 読み込んだファイルの情報を取得
        number_of_cycles = df["FP"].max()  # ファイル内の周期数

        # 無次元化
        # 変数の定義
        n_points = len(t_interp)  # 出力の点数．間隔の数はこれ-1になる．  # ここにアンサンブル平均を求めたい列名のリストを定義します

        n_cols = 2  # 1行に表示するグラフの数
        n_rows = (len(DATA_COLUMN_NAMES) + n_cols - 1) // n_cols  # 必要な行数

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))  # サブプロットを作成
        axs = axs.flatten()  # 2次元配列を1次元配列に変換

        for idx, column_name in enumerate(DATA_COLUMN_NAMES):
            # y_interp の列数を df["FP"].max() + 1 に合わせる
            y_interp = np.zeros((n_points, int(number_of_cycles + 1)))

            # 周期ごとに繰り返し計算する
            # 線形補完
            for i in range(int(df["FP"].max() + 1)):
                # この周期のデータ
                one_cycle_data = df.loc[df["FP"] == i, :]

                t_input = np.linspace(1, len(one_cycle_data), len(one_cycle_data))
                t_input = (t_input - 1) / (len(one_cycle_data) - 1)

                y_input = one_cycle_data[column_name].copy().values

                # 無次元化
                y_interp[:, i] = np.interp(t_interp, t_input, y_input)

            # アンサンブル平均計算
            y_ensemble_average = np.mean(y_interp, axis=1)  # アンサンブル平均
            y_SD = np.std(y_interp, axis=1)  # 生データの標準偏差

            if "EN" in column_name:
                color = EN_COLORS[int(column_name[2:]) - 1]
                if en_y_lim:
                    axs[idx].set_ylim(en_y_lim)

            elif "SG" in column_name:
                color = SG_COLORS[int(column_name[2:]) - 1]
                if sn_y_lim:
                    axs[idx].set_ylim(sn_y_lim)
            else:
                color = "r"

            axs[idx].plot(t_interp, y_ensemble_average, color, label=column_name)
            axs[idx].fill_between(t_interp, y_ensemble_average + y_SD, y_ensemble_average - y_SD,
                                  facecolor=color, alpha=0.2)

            # サブプロットに描画
            axs[idx].set_xlabel("Normalized time (t/T)")
            axs[idx].set_ylabel("Strain (μSTR)")
            axs[idx].set_xlim((0, 1))
            axs[idx].legend(loc="best", fontsize=15)

        # 不要なサブプロットを削除
        for idx in range(len(DATA_COLUMN_NAMES), n_cols * n_rows):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        # plt.show()

        return fig
