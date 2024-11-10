from datetime import timedelta, datetime
from glob import glob
import random
from time import time

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tqdm.notebook import tqdm

from kubota_libraries.general_functions import *
from kubota_libraries.models import *
from settings.general_variables import *

random.seed(40)
np.random.seed(40)
tf.random.set_seed(40)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)



# カスタムコールバックの定義
class SimpleTqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.prog_bar = None

    def on_train_begin(self, logs=None):
        self.prog_bar = tqdm(total=self.total_epochs, desc="Epoch Progress", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.prog_bar.update(1)

    def on_train_end(self, logs=None):
        self.prog_bar.close()

class BaseCNNModel:

    def __init__(self, test_size, random_state):
        self.test_size = test_size
        self.random_state = random_state

        # 初期化
        self.should_use_flatten_data = False
        self.should_normalize = False
        self.which_channel = None

        self.chosen_parameter_folder_name = ""

        self.training_data_path = ""
        self.model_path = ""

        self.input_train = None
        self.input_test = None
        self.answer_train = None
        self.answer_test = None

        self.model = None
        self.history = None
        self.accuracy_num = None
        self.accuracy_percentage = None

    def _get_training_data_path(self):
        """
        学習データの保存先のパスを取得します。
        Returns:
            str: 学習データの保存先のパス
        """
        training_data_path = TRAINING_FILE_PATH

        if self.should_use_flatten_data:
            training_data_path += f"/{FLATTEN_DATA_FOLDER_NAME}"
            if self.m and self.frequency_c:
                training_data_path += f"/{self.chosen_parameter_folder_name}/cutoff{self.frequency_c}_m{self.m}"
            else:
                raise ValueError("mとcutoff frequencyを定義してください。")
        else:
            training_data_path += f"/{RAW_DATA_FOLDER_NAME}/{self.chosen_parameter_folder_name}"
        # else:
        #     raise Exception("Stop it.")

        if self.should_normalize:
            training_data_path += f"/{NORMALIZED_DATA_FOLDER_NAME}"
        else:
            training_data_path += f"/{NOT_NORMALIZED_DATA_FOLDER_NAME}"

        return training_data_path

    def _get_model_path(self):
        """
        モデルの保存先のパスを取得します。
        Returns:
            str: モデルの保存先のパス
        """

        model_path = MODEL_FILE_PATH

        if self.should_use_flatten_data:
            model_path += f"/{FLATTEN_DATA_FOLDER_NAME}"
        else:
            model_path += f"/{RAW_DATA_FOLDER_NAME}"
        # else:
        #     raise Exception("Stop it.")

        if self.should_normalize:
            model_path += f"/{self.chosen_parameter_folder_name}"
        else:
            model_path += f"/{self.chosen_parameter_folder_name}"

        if self.m and self.frequency_c:
            print(f"/cutoff{self.frequency_c}_m{self.m}")
            model_path += f"/cutoff{self.frequency_c}_m{self.m}"
        else:
            raise ValueError("mとfrequency_cを定義してください。")

        if self.should_normalize:
            model_path += f"/{NORMALIZED_DATA_FOLDER_NAME}"
        else:
            model_path += f"/{NOT_NORMALIZED_DATA_FOLDER_NAME}"


        return model_path

    def _get_model_result_path(self):
        """
        モデルの性能診断結果の保存先のパスを取得します。
        性能診断結果は、ここでは、正答率、history等をさします。
        Returns:
            str: モデルの保存先のパス
        """

        model_result_path = self.model_path

        if self.which_channel:
            model_result_path += \
                f"/{MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}_{MODEL_RESULT_FOLDER_SUFFIX}"
        else:
            model_result_path += f"/{MODEL_FILE_PREFIX}_{MODEL_RESULT_FOLDER_SUFFIX}"

        create_directory(model_result_path)
        return model_result_path

    def _print_current_settings(self):
        """
        現在の設定を出力します。
        """
        print("Should Flatten: ", self.should_use_flatten_data)
        chosen_parameter_folder_names_list = self.chosen_parameter_folder_name.split("/")
        print("Split parameter: ", chosen_parameter_folder_names_list[0])
        if len(chosen_parameter_folder_names_list) == 2:
            print("Flattening parameter: ", chosen_parameter_folder_names_list[1])
        if self.which_channel:
            print("Which Channel: ", " & ".join([DATA_COLUMN_NAMES[int(x) - 1] for x in self.which_channel]))
        else:
            print("Which Channel: ", "All")
        print("Should Normalize: " + str(self.should_normalize))
        return

    def _print_current_data(self):
        """
        現在のデータを出力します。
        """
        if self.input_test is not None:
            num_of_train_data_per_class = [f"{wind_environment}：{np.count_nonzero(self.answer_train == i)}"
                                           for i, wind_environment in enumerate(WIND_ENVIRONMENTS)]
            num_of_test_data_per_class = [f"{wind_environment}：{np.count_nonzero(self.answer_test == i)}"
                                          for i, wind_environment in enumerate(WIND_ENVIRONMENTS)]

            print(f"Num of train data：{len(self.answer_train)} ({' / '.join(num_of_train_data_per_class)})")
            print(f"Num of test data：{len(self.answer_test)} ({' / '.join(num_of_test_data_per_class)})")
        else:
            print(IMPLEMENTATION_LOAD_DATA_WARNING)
        return

    def _reset_data(self):
        """
        データを削除し、初期状態に戻します。
        """
        del self.input_train, self.input_test, self.answer_train, self.answer_test

        self.input_train = None
        self.input_test = None
        self.answer_train = None
        self.answer_test = None
        return

    def _reset_model(self):
        """
        モデルを削除し、初期状態に戻します。
        """
        del self.model, self.history, self.accuracy_percentage, self.accuracy_num

        self.model = None
        self.history = None
        self.accuracy_num = None
        self.accuracy_percentage = None
        return

    def _reset_path(self):
        """
        現在のpathを削除し、初期状態に戻します。
        """
        del self.chosen_parameter_folder_name, self.training_data_path, self.model_path

        self.chosen_parameter_folder_name = ""

        self.training_data_path = ""
        self.model_path = ""
        return

    def _update_settings(self, root_folder):
        """
        各種設定を行います。
        Arg:
            root_folder: rootのfolder
        """

        self._reset_path()

        self.should_use_flatten_data = input(QUESTION_PROMPT_FLATTEN_OR_RAW) == "1"

        interaction_current_path = root_folder

        if self.should_use_flatten_data:
            interaction_current_path += f"/{FLATTEN_DATA_FOLDER_NAME}"
        else:
            interaction_current_path += f"/{RAW_DATA_FOLDER_NAME}"

        data_choices_split_num = [os.path.basename(path) for path in sorted(glob(f"{interaction_current_path}/*"))]

        interaction_current_path = self._user_data_selection(data_choices_split_num, QUESTION_PROMPT_SPLIT_NUM,
                                                             interaction_current_path)

        if self.should_use_flatten_data:
            data_choices_flatten_parameter = [os.path.basename(path) for path in
                                              sorted(glob(f"{interaction_current_path}/*"))]
            _ = self._user_data_selection(data_choices_flatten_parameter,
                                          QUESTION_PROMPT_FLATTENING_PARAMETER,
                                          interaction_current_path)

        self.chosen_parameter_folder_name = self.chosen_parameter_folder_name[1:]

        return interaction_current_path

    def _user_data_selection(self, data_choices, prompt_message, interaction_current_path):
        while True:
            question = f"{prompt_message}\n\n"
            for i, data_choice in enumerate(data_choices):
                question += f"{i + 1}: {data_choice}\n"
            question += f"\n{QUESTION_PROMPT_END}"

            picked = input(question)

            if picked.lower() == "end":
                print(NO_FILE_CHOSEN_WARNING)
                return None

            if int(picked) <= len(data_choices):
                self.chosen_parameter_folder_name += f"/{data_choices[int(picked) - 1]}"
                interaction_current_path += self.chosen_parameter_folder_name
                return interaction_current_path

    def evaluate_average_inference_time(self, warmup_rounds=2, measure_rounds=10):
        """
        モデルの推論にかかる平均時間を計測します。初期の数回（デフォルトでは2回）の推論は
        モデルやハードウェアのウォームアップに使用され、計測から除外されます。その後、
        指定された回数（デフォルトでは10回）の推論を行い、それらの推論にかかる時間の平均を計算します。

        Args:
            warmup_rounds : ウォームアップのラウンド数 (デフォルト: 2)
            measure_rounds : 推論時間を計測するラウンド数 (デフォルト: 10)

        Returns:
            average_time : 推論にかかる平均時間（秒）
        """

        # ウォームアップラウンド
        for _ in range(warmup_rounds):
            _ = self.model.predict(self.input_test)

        # 実際の計測
        times = []
        for _ in range(measure_rounds):
            start_time = time()
            _ = self.model.predict(self.input_test)
            end_time = time()
            times.append(end_time - start_time)

        average_time = np.mean(times)

        return average_time

    def evaluate_model_flops(self):
        """
        モデルのFLOPS（Floating Point Operations Per Second）を計測します。この関数は、
        モデルが1秒間に実行する浮動小数点演算の数を推定し、モデルの計算効率を評価します。
        FLOPSの高いモデルはより多くの演算を高速に処理できることを意味しますが、
        通常、より多くのエネルギー消費やハードウェアリソースを必要とします。

        Returns:
            flops : モデルのFLOPS
        """

        concrete = tf.function(lambda inputs: self.model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in self.model.inputs])
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

        # FLOPSを計算
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops

    def evaluate_model_memory_usage(self):
        # モデルを実行する前のメモリ使用量を取得
        initial_memory = psutil.Process().memory_info().rss

        # モデルを実行
        _ = self.model.predict(self.input_test)

        # モデルを実行した後のメモリ使用量を取得
        final_memory = psutil.Process().memory_info().rss

        # 使用されたメモリ量を計算
        memory_used = final_memory - initial_memory
        return memory_used

    def evaluate_model(self, measure_inference_time=True, measure_memory_usage=True, measure_flops=True):
        results = []
        columns = ['Metric', 'Value']

        # Evaluate model accuracy
        test_result = self.model.evaluate(self.input_test, self.answer_test)
        results.append(['Test Accuracy', f"{test_result[1] * 100} %"])

        # Measure inference time
        if measure_inference_time:
            average_time = self.evaluate_average_inference_time()
            value, prefix = format_number(average_time)
            results.append(['Estimation Time', f"{value} {prefix}s"])

        # Measure memory usage
        if measure_memory_usage:
            memory_used = self.evaluate_model_memory_usage()
            value, prefix = format_number(memory_used)
            results.append(['Memory used by the model', f"{value} {prefix}B"])

        # Measure FLOPS
        if measure_flops:
            flops = self.evaluate_model_flops()
            value, prefix = format_number(flops)
            results.append(['Model FLOPS', f"{value} {prefix}FLOP/s"])

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results, columns=columns)

        return results_df

    def load_train_test_data(self):
        """
        訓練データとテストデータを取得します。
        Returns:
            numpy array: 訓練データとテストデータ（入力とラベル）
        """

        self._reset_data()

        input_datasets = np.load(f"{self.training_data_path}/input_datasets.npy")
        answer_datasets = np.load(f"{self.training_data_path}/answer_datasets.npy")

        print(self.which_channel)
        if self.which_channel == "":
            selected_columns = [i for i in range(len(DATA_COLUMN_NAMES))]
        else:
            selected_columns = [int(self.which_channel[i]) - 1 for i in range(len(self.which_channel))]
        # else:
        #     raise ValueError(f"Invalid which_channel value: {self.which_channel}")

        input_datasets = input_datasets[:, :, selected_columns]

        self.input_train, self.input_test, self.answer_train, self.answer_test = train_test_split(
            input_datasets, answer_datasets,
            test_size=self.test_size, random_state=self.random_state, stratify=answer_datasets
        )

        self._print_current_settings()
        self._print_current_data()
        return

    def optimal_epochs(self):
        """
        historyを使って、モデルが最適な状態の時のインデックス（epochs）を出力する。
        最適な状態は以下の通り
        ・loss, val_lossが最小の時のepochs（index+1)とその時のloss, val_lossの値
        ・accuracy, val_accuracyが最小の時のepochs（index+1)とその時のaccuracy, val_accuracyの値
        """
        # lossとval_lossの最小値を見つけ、そのインデックス（epoch）を取得
        min_loss_epoch = self.history['loss'].idxmin() + 1
        min_val_loss_epoch = self.history['val_loss'].idxmin() + 1

        # accuracyとval_accuracyの最大値を見つけ、そのインデックス（epoch）を取得
        max_accuracy_epoch = self.history['accuracy'].idxmax() + 1
        max_val_accuracy_epoch = self.history['val_accuracy'].idxmax() + 1

        # データフレームを作成
        df = pd.DataFrame({
            'Metric': ['loss', 'val_loss', 'accuracy', 'val_accuracy'],
            'Optimal Epoch': [min_loss_epoch, min_val_loss_epoch, max_accuracy_epoch, max_val_accuracy_epoch],
            'Value at Optimal Epoch': [
                self.history['loss'].iloc[min_loss_epoch - 1],
                self.history['val_loss'].iloc[min_val_loss_epoch - 1],
                self.history['accuracy'].iloc[max_accuracy_epoch - 1],
                self.history['val_accuracy'].iloc[max_val_accuracy_epoch - 1]
            ]
        })

        return df

    def plot_history(self):
        """
        historyを折れ線グラフで表示する。
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # loss, val_lossをプロット
        ax1.plot(self.history.index + 1, self.history['loss'], label='loss')
        ax1.plot(self.history.index + 1, self.history['val_loss'], label='val_loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_ylim(bottom=0)  # 表示するyの範囲は0以上
        ax1.set_xlim(left=1, right=self.history.index.max() + 1)  # 表示するxの範囲は1からhistory.index + 1の最大値
        ax1.grid(True)  # グリッドを追加

        # accuracy, val_accuracyをプロット
        ax2.plot(self.history.index + 1, self.history['accuracy'], label='accuracy')
        ax2.plot(self.history.index + 1, self.history['val_accuracy'], label='val_accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.set_ylim(0, 1)  # 表示するyの範囲は常に0から1まで
        ax2.set_xlim(left=1, right=self.history.index.max() + 1)  # 表示するxの範囲は1からhistory.index + 1の最大値
        ax2.grid(True)  # グリッドを追加

        plt.tight_layout()  # レイアウトの調整
        return fig

    def visualize_confusion_matrix(self, output_type="num"):
        """
        予測結果の混同行列を視覚化し、オプションでCSV形式で保存します。

        Args:
            output_type (str): 混同行列の出力形式。"num"は実際の数値、"percentage"はパーセンテージを示します。

        Returns:
            fig (matplotlib.figure.Figure): 混同行列の視覚化を含むmatplotlib Figure オブジェクトを返します。
        """

        # テストデータを使って予測を行う
        predictions = self.model.predict(self.input_test)
        predictions = np.argmax(predictions, axis=1)  # 最も確率が高いクラスを選択

        # 混同行列を計算する
        cm = confusion_matrix(self.answer_test, predictions)

        # 混同行列を正規化する（割合を計算する）
        if output_type == "percentage":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 混同行列をPandasのDataFrameに変換する
        cm_df = pd.DataFrame(cm, index=WIND_ENVIRONMENTS, columns=WIND_ENVIRONMENTS)

        if output_type == "percentage":
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))
            self.accuracy_percentage = cm_df
            # フォーマットを小数点以下2桁にし、カラーマップを'Reds'に設定
            sns.heatmap(self.accuracy_percentage, annot=True, fmt=".2f", cmap='Reds', ax=ax1)
        else:
            fig, axs = plt.subplots(2, 1, figsize=(15, 12),
                                    gridspec_kw={'height_ratios': [len(cm_df), 1]})
            ax1, ax2 = axs
            self.accuracy_num = cm_df
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Reds", ax=ax1)
        ax1.set_title('Confusion matrix')
        ax1.set_ylabel('Actual label')
        ax1.set_xlabel('Predicted label')

        # Add the column sum of the confusion matrix
        if output_type == "num":
            cm_sum = cm_df.sum(axis=0)
            cm_sum_df = pd.DataFrame(cm_sum).transpose()
            sns.heatmap(cm_sum_df, annot=True, fmt="d", cmap="Blues", ax=ax2)
            ax2.set_ylabel('Sum of prediction')
            ax2.set_xlabel('Predicted label')

        plt.tight_layout()
        plt.show()

        return fig


class CNNModelTrainer(BaseCNNModel):
    """
    CNNモデルを訓練するクラス。
    Args:
        test_size: テストデータの割合（デフォルトは0.20）
        random_state: 乱数の種（デフォルトは40）
    """

    def __init__(self, test_size, random_state):

        super().__init__(test_size, random_state)
        self.m = None
        self.frequency_c = None

    def __append_history(self, new_history):
        """
        既存のhistoryに新しいhistoryを追加するクラスメソッド。

        Args:
            new_history (dict): 追加する新しいhistory。'loss', 'accuracy'などのキーと、そのエポックごとの値のリストを持つ辞書。
        """
        new_history_df = pd.DataFrame(new_history)

        if self.history is None:
            self.history = new_history_df
        else:
            self.history = pd.concat([self.history, new_history_df]).reset_index(drop=True)

        return

    def __save_model_history(self):
        """
        現在のモデルのhistoryを保存します。
        """
        model_result_path = self._get_model_result_path()

        if self.which_channel:
            self.history.to_csv(
                f"{model_result_path}/{MODEL_HISTORY_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}.csv")
        else:
            self.history.to_csv(f"{model_result_path}/{MODEL_HISTORY_FILE_PREFIX}.csv")
        return

    def __select_model_based_on_data(self):
        """
        テストデータの長さに基づき、適切なモデルを選択する。

        Returns:
            選択されたモデル（tensorflow.keras.models.Model）
        """

        input_train_shape = self.input_train.shape

        if MODEL_FILE_NAME == "1_CNN":

            test_data_length = self.input_train.shape[1]
            if test_data_length > 512 * 2:
                return create_model_210720([512, 256, 128, 64, 32, 16], input_train_shape)
            elif test_data_length > 256 * 2:
                return create_model_210720([256, 128, 64, 32, 16], input_train_shape)
            elif test_data_length > 128 * 2:
                return create_model_210720([128, 64, 32, 16], input_train_shape)
            elif test_data_length > 64 * 2:
                return create_model_210720([64, 32, 16], input_train_shape)
            elif test_data_length > 32 * 2:
                return create_model_210720([32, 16], input_train_shape)
            elif test_data_length > 16 * 2:
                return create_model_210720([16], input_train_shape)
            else:
                raise ValueError("Your data is too small.")

        elif MODEL_FILE_NAME == "2_SIMPLE_NN":
            return create_simple_dense_model_231110(input_train_shape)

    def check_folders_existence(self, should_use_flatten_data_list, should_normalize_list, which_channels_list,
                                chosen_parameter_folder_names_list, m_and_cfs=[{}]):
        """
        指定されたパラメータの組み合わせごとに、訓練データのフォルダが存在するかどうかをチェックする関数。

        Args:
            should_use_flatten_data_list (list): データをフラット化するかどうかを指定するブール値のリスト。
            should_normalize_list (list): データを正規化するかどうかを指定するブール値のリスト。
            which_channels_list (list): 使用するチャンネルを指定する文字列のリスト。
            chosen_parameter_folder_names_list (list): パラメータを分割するために選択されたフォルダ名のリスト。

        Returns:
            None. 存在しないフォルダがある場合はその情報を出力し、全てのフォルダが存在する場合はその旨を出力する。

        Note:
            各リストの全ての組み合わせについてチェックを行う。各組み合わせごとに設定を更新し、対応する訓練データのフォルダの存在を確認する。
        """
        not_found_folders = ["Folders below does not exist."]

        for should_use_flatten_data in should_use_flatten_data_list:
            for should_normalize in should_normalize_list:
                for which_channel in which_channels_list:
                    for chosen_parameter_folder_name in chosen_parameter_folder_names_list:
                        for m_and_cf in m_and_cfs:
                            self.update_settings_efficiently(
                                should_use_flatten_data=should_use_flatten_data,
                                should_normalize=should_normalize,
                                chosen_parameter_folder_name=chosen_parameter_folder_name,
                                which_channel=which_channel,
                                m_and_cf=m_and_cf
                            )

                            if os.path.isdir(self.training_data_path):
                                pass
                            else:
                                file_info = "=================================================\n"
                                file_info += f"Should Flatten: {should_use_flatten_data}\n"

                                chosen_parameter_folder_names_list = self.chosen_parameter_folder_name.split("/")
                                file_info += f"Split parameter: {chosen_parameter_folder_names_list[0]}"
                                if len(chosen_parameter_folder_names_list) == 2:
                                    file_info += f"Flattening parameter: {chosen_parameter_folder_names_list[1]}"

                                if which_channel:
                                    file_info += f"Which Channel: {which_channel}\n"
                                else:
                                    file_info += "Which Channel: All"

                                file_info += f"Should Normalize: {should_normalize}\n"
                                file_info += "================================================="
                                not_found_folders.append(file_info)

        if len(not_found_folders) > 1:
            raise FileNotFoundError("\n".join(not_found_folders))
        else:
            print("All folders exist.")
        return

    def get_model(self, optimizer, loss, metrics):
        """
        モデルを作成します。
        Args:
            optimizer:
            loss:
            metrics:
        """
        self._reset_model()

        self.model = self.__select_model_based_on_data()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return self.model.summary()

    def save_model(self):
        """
        現在のモデルを保存します。
        """
        print(self.model_path)

        if self.which_channel:
            self.model.save(f"{self.model_path}/{MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}")
            self.model.save_weights(
                f"{self.model_path}/{MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}.h5")
        else:
            print(f"{self.model_path}/{MODEL_FILE_PREFIX}")
            self.model.save(f"{self.model_path}/{MODEL_FILE_PREFIX}")
            self.model.save_weights(f"{self.model_path}/{MODEL_FILE_PREFIX}.h5")

        self.__save_model_history()
        return

    def train_model(self, epochs, verbose, validation_split, batch_size, mode="manual"):
        """
        モデルを訓練します。
        Args:
            epochs:
            verbose:
            validation_split:
            batch_size:
            mode:
        """

        tensorboard = TensorBoard(log_dir=f"{TENSORFLOW_LOG_FILE_PATH}/{time()}")

        if self.which_channel:
            best_model_path = f"{self.model_path}/{BEST_MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}.h5"
        else:
            best_model_path = f"{self.model_path}/{BEST_MODEL_FILE_PREFIX}.h5"
        model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', mode='max',
                                           verbose=0, save_best_only=True)

        callbacks = [tensorboard, model_checkpoint]
        if verbose == 0:
            simple_tqdm_callback = SimpleTqdmCallback(total_epochs=epochs)
            callbacks.append(simple_tqdm_callback)

        start_time = time()
        print(f"Training start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

        history = self.model.fit(self.input_train, self.answer_train, epochs=epochs, verbose=verbose,
                                 validation_split=validation_split, batch_size=batch_size, shuffle=True,
                                 callbacks=callbacks)

        end_time = time()
        print(f"Training end time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")

        training_time = end_time - start_time
        training_time_str = str(timedelta(seconds=training_time))

        if mode == "efficient":
            print()
            self._print_current_settings()
        print(f'Training took {training_time_str} (h:m:s)')

        self.__append_history(history.history)

        return

    def update_settings(self):
        """
        各種設定をユーザーとインタラクティブに行います。
        """

        _ = super()._update_settings(TRAINING_FILE_PATH)

        while True:
            self.which_channel = input(QUESTION_PROMPT_WHICH_COLUMN)
            if all(1 <= int(x) <= len(DATA_COLUMN_NAMES) for x in self.which_channel):
                break
            print(f"1-{len(DATA_COLUMN_NAMES)}で選んでください")

        self.training_data_path = self._get_training_data_path()
        self.should_normalize = input(QUESTION_PROMPT_NORMALIZE_ML_DATA).lower() == "y"
        self.model_path = self._get_model_path()
        self._print_current_settings()

        return

    def update_settings_efficiently(self, should_use_flatten_data, should_normalize, chosen_parameter_folder_name,
                                    which_channel, m_and_cf={}):
        """
        各種設定を変数を用いてより効率的に行います。
        繰り返し文等で、複数の設定による学習を行いたいときに利用します。
        """
        if should_use_flatten_data:
            self.should_use_flatten_data = should_use_flatten_data
            if m_and_cf:
                self.m = m_and_cf["m"]
                self.frequency_c = m_and_cf["frequency_c"]

        self.should_normalize = should_normalize
        self.chosen_parameter_folder_name = chosen_parameter_folder_name
        self.which_channel = which_channel

        self.training_data_path = self._get_training_data_path()
        self.model_path = self._get_model_path()
        return


class CNNModelUser(BaseCNNModel):

    def __init__(self, test_size, random_state):
        super().__init__(test_size, random_state)

    def check_folders_existence(self, should_use_flatten_data_list, should_normalize_list, which_channels_list,
                                chosen_parameter_folder_names_list, m_and_cfs=[]):
        """
        指定されたパラメータの組み合わせごとに、訓練データのフォルダが存在するかどうかをチェックする関数。

        Args:
            should_use_flatten_data_list (list): データをフラット化するかどうかを指定するブール値のリスト。
            should_normalize_list (list): データを正規化するかどうかを指定するブール値のリスト。
            which_channels_list (list): 使用するチャンネルを指定する文字列のリスト。
            chosen_parameter_folder_names_list (list): パラメータを分割するために選択されたフォルダ名のリスト。

        Returns:
            None. 存在しないフォルダがある場合はその情報を出力し、全てのフォルダが存在する場合はその旨を出力する。

        Note:
            各リストの全ての組み合わせについてチェックを行う。各組み合わせごとに設定を更新し、対応する訓練データのフォルダの存在を確認する。
        """
        not_found_folders = ["Folders below does not exist."]

        for should_use_flatten_data in should_use_flatten_data_list:
            for should_normalize in should_normalize_list:
                for which_channel in which_channels_list:
                    for chosen_parameter_folder_name in chosen_parameter_folder_names_list:
                        for m_and_cf in m_and_cfs:
                            self.update_settings_efficiently(
                                should_use_flatten_data=should_use_flatten_data,
                                should_normalize=should_normalize,
                                chosen_parameter_folder_name=chosen_parameter_folder_name,
                                which_channel=which_channel,
                                m_and_cf=m_and_cf
                            )

                            if os.path.isdir(self.model_path):
                                pass
                            else:
                                file_info = "=================================================\n"
                                file_info += f"Should Flatten: {should_use_flatten_data}\n"

                                chosen_parameter_folder_names_list = self.chosen_parameter_folder_name.split("/")
                                file_info += f"Split parameter: {chosen_parameter_folder_names_list[0]}"
                                if len(chosen_parameter_folder_names_list) == 2:
                                    file_info += f"Flattening parameter: {chosen_parameter_folder_names_list[1]}"

                                if which_channel:
                                    file_info += f"Which Channel: {which_channel}\n"
                                else:
                                    file_info += "Which Channel: All"

                                file_info += f"Should Normalize: {should_normalize}\n"
                                file_info += "================================================="
                                not_found_folders.append(file_info)

        if len(not_found_folders) > 1:
            raise FileNotFoundError("\n".join(not_found_folders))
        else:
            print("All folders exist.")
        return

    def effciently_evaluate_models(self, should_use_flatten_data_list, should_normalize_list, which_channels_list,
                                   chosen_parameter_folder_names_list, m_and_cfs=[{}]):
        combined_results = pd.DataFrame(
            columns=["Metric", "Test Accuracy", "Estimation Time", "Memory used by the model", "Model FLOPS",
                     "Should Flatten", "Split Parameter", "Which Channel", "Should Normalize"])
        for should_use_flatten_data in should_use_flatten_data_list:
            for should_normalize in should_normalize_list:
                for which_channel in which_channels_list:
                    for chosen_parameter_folder_name in chosen_parameter_folder_names_list:
                        for m_and_cf in m_and_cfs:
                            self.update_settings_efficiently(
                                should_use_flatten_data=should_use_flatten_data,
                                should_normalize=should_normalize,
                                chosen_parameter_folder_name=chosen_parameter_folder_name,
                                which_channel=which_channel,
                                m_and_cf=m_and_cf
                            )
                            self.load_train_test_data()
                            self.load_saved_model(get_the_best=True)
                            tmp_result = self.evaluate_model()

                            if self.which_channel:
                                displayed_which_channel = " & ".join(
                                    [DATA_COLUMN_NAMES[int(x) - 1] for x in self.which_channel])
                            else:
                                displayed_which_channel = "All"

                            result_config = {
                                "Should Flatten": should_use_flatten_data,
                                "m": self.m,
                                "Cutoff frequency": self.frequency_c,
                                "Split Parameter": chosen_parameter_folder_name,
                                "Which Channel": displayed_which_channel,
                                "Should Normalize": should_normalize,
                            }

                            tmp_result = tmp_result.set_index("Metric").T
                            tmp_result = tmp_result.assign(**result_config).reset_index(drop=True)

                            combined_results = pd.concat([combined_results, tmp_result]).reset_index(drop=True)
        return combined_results.drop("Metric", axis=1)

    def load_history(self):
        """
        対象モデルの学習historyをロードします。
        """
        model_result_path = self._get_model_result_path()

        if self.which_channel:
            self.history = pd.read_csv(
                f"{model_result_path}/{MODEL_HISTORY_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}.csv")
        else:
            self.history = pd.read_csv(f"{model_result_path}/{MODEL_HISTORY_FILE_PREFIX}.csv")
        return

    def load_saved_model(self, get_the_best):
        """
        保存されたモデルをロードします。
        Returns:
            keras Model: 保存されたモデル
        """

        if self.which_channel:
            reconstruct_model = keras.models.load_model(
                f"{self.model_path}/{MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}")
            if get_the_best:
                reconstruct_model.load_weights(
                    f"{self.model_path}/{BEST_MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}.h5")
            else:
                reconstruct_model.load_weights(
                    f"{self.model_path}/{MODEL_FILE_PREFIX}_{CHANNEL_FILE_PREFIX}{self.which_channel}.h5")
        else:
            reconstruct_model = keras.models.load_model(f"{self.model_path}/{MODEL_FILE_PREFIX}")
            if get_the_best:
                reconstruct_model.load_weights(f"{self.model_path}/{BEST_MODEL_FILE_PREFIX}.h5")
            else:
                reconstruct_model.load_weights(f"{self.model_path}/{MODEL_FILE_PREFIX}.h5")

        self.model = reconstruct_model
        return

    def save_accuracy(self):
        """
        回答の正答率をcsv形式で保存します。
        """

        # 混同行列のデータフレームをCSV形式で保存する
        model_result_path = self._get_model_result_path()

        if self.which_channel:
            self.accuracy_num.to_csv(
                f"{model_result_path}/{MODEL_ACCURACY_FILE_PREFIX}_{MODEL_ACCURACY_FILE_NUM_SUFFIX}.csv")
            self.accuracy_percentage.to_csv(
                f"{model_result_path}/{MODEL_ACCURACY_FILE_PREFIX}_{MODEL_ACCURACY_FILE_PERCENTAGE_SUFFIX}.csv")
        else:
            self.accuracy_num.to_csv(
                f"{model_result_path}/{MODEL_ACCURACY_FILE_PREFIX}_{MODEL_ACCURACY_FILE_NUM_SUFFIX}.csv")
            self.accuracy_percentage.to_csv(
                f"{model_result_path}/{MODEL_ACCURACY_FILE_PREFIX}_{MODEL_ACCURACY_FILE_PERCENTAGE_SUFFIX}.csv")
        return

    def update_settings(self):
        """
        各種設定を行います。
        """

        interaction_current_path = super()._update_settings(MODEL_FILE_PATH)

        model_choices_normalization = [os.path.basename(path) for path in
                                       sorted(glob(f"{interaction_current_path}/*"))]
        if len(model_choices_normalization) == 1:
            if NORMALIZED_DATA_FOLDER_NAME in model_choices_normalization:
                self.should_normalize = True
            elif NOT_NORMALIZED_DATA_FOLDER_NAME in model_choices_normalization:
                self.should_normalize = False
            print(f"Only {model_choices_normalization[0]} was found.")
        else:
            while True:
                should_use_normalized_data_model = input(
                    f"Which model would you like to use?"
                    f"\n1:{NORMALIZED_DATA_FOLDER_NAME} model\n2:{NOT_NORMALIZED_DATA_FOLDER_NAME} model"
                )
                if should_use_normalized_data_model == "1":
                    self.should_normalize = True
                    break
                elif should_use_normalized_data_model == "2":
                    self.should_normalize = False
                    break

        if self.should_normalize:
            interaction_current_path += "/normalized"
        else:
            interaction_current_path += "/not_normalized"

        # フォルダ名を取得し、選択肢として適切なものを絞り込む
        base_folder_path_names = [os.path.basename(path) for path in sorted(glob(f"{interaction_current_path}/*"))]
        model_choices_channels = [name for name in base_folder_path_names if not any(
            phrase in name for phrase in [BEST_MODEL_FILE_PREFIX, MODEL_RESULT_FOLDER_SUFFIX, ".h5"])]

        # ユーザーからの入力を受け取り、適切な選択を行うまで繰り返す
        while True:
            question = f"{QUESTION_PROMPT_WHICH_CHANNEL}\n\n"
            question += "\n".join(f"{i + 1}: {choice}" for i, choice in enumerate(model_choices_channels))
            question += f"\n{QUESTION_PROMPT_END}"

            picked = input(question)

            if picked.lower() == "end":
                print(NO_FILE_CHOSEN_WARNING)
                break

            picked_index = int(picked) - 1
            if 0 <= picked_index < len(model_choices_channels):
                picked_channel = model_choices_channels[picked_index]
                self.which_channel = picked_channel.split(CHANNEL_FILE_PREFIX)[
                    -1] if CHANNEL_FILE_PREFIX in picked_channel else ""
                break

        self.training_data_path = self._get_training_data_path()
        self.model_path = self._get_model_path()

        self._print_current_settings()

        return

    def update_settings_efficiently(self, should_use_flatten_data, should_normalize, chosen_parameter_folder_name,
                                    which_channel, m_and_cf={}):
        """
        各種設定を変数を用いてより効率的に行います。
        繰り返し文等で、複数の設定による学習を行いたいときに利用します。
        """
        self.should_normalize = should_normalize
        self.chosen_parameter_folder_name = chosen_parameter_folder_name
        self.which_channel = which_channel
        if should_use_flatten_data:
            self.should_use_flatten_data = should_use_flatten_data
            if m_and_cf:
                self.m = m_and_cf["m"]
                self.frequency_c = m_and_cf["frequency_c"]


        self.training_data_path = self._get_training_data_path()
        self.model_path = self._get_model_path()
        return
