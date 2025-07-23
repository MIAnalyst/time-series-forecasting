import pandas as pd
import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers  
import psutil
import os
import matplotlib.pyplot as plt
from functools import reduce
import random
from keras_tuner import RandomSearch

def set_seeds(seed=42):
    # Ensures deterministic behavior for operations that rely on Python's internal hashing (e.g., set, dict, some data loading).
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds(42)



class DataWindow:
    """
    A utility class for preparing and managing windowed time series data for sequence models.

    This class handles:
    - Splitting time series data into input and label windows.
    - Separating numerical features from categorical ones (e.g., CountryCode).
    - Generating input tensors for models that use numerical and embedding inputs.
    - Visualizing sample input/label/prediction windows.

    Parameters:
    ----------
    input_width : int
        Number of timesteps used as input.
    label_width : int
        Number of timesteps to predict as output.
    shift : int
        Offset between the end of the input window and the start of the label window.
    train_df : pd.DataFrame
        DataFrame for training data.
    val_df : pd.DataFrame
        DataFrame for validation data.
    test_df : pd.DataFrame
        DataFrame for test data.
    label_columns : list of str, optional
        Names of the columns to be predicted (target variables). If None, all columns are used.
    shuffle : bool, default=True
        Whether to shuffle training examples during training.

    Attributes:
    ----------
    column_indices : dict
        Mapping of column names to their index positions.
    label_columns_indices : dict
        Mapping of label column names to their index positions.
    total_window_size : int
        Total size of the input + shift window.
    input_slice : slice
        Slice object defining the input portion of the window.
    labels_slice : slice
        Slice object defining the label portion of the window.
    input_indices : np.ndarray
        Timestep indices for input window.
    label_indices : np.ndarray
        Timestep indices for label window.
    """
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None,shuffle=True):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.shuffle = shuffle

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    
    def split_to_inputs_labels(self, features):
        "Splits a windowed tensor batch into model inputs and target labels"
        # (batch size, sequence length, num features)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # CountryCode is split from the numerical features and provided separately so the 
        # model can use an embedding layer to learn a dense, learnable representation of 
        # each country, while the numerical features are used directly as continuous inputs for forecasting.
        country_idx = self.column_indices['CountryCode']
        country = tf.cast(inputs[:, :, country_idx], tf.int32)
        float_indices = [i for i in range(features.shape[2]) if i != country_idx]
        numerical = tf.gather(inputs, float_indices, axis=-1)

        # This code keeps only the target column(s) in the labels tensor.
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        numerical.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        country.set_shape([None, self.input_width])

        return {"numerical": numerical, "country": country}, labels

    def plot(self, model=None, plot_col=None, max_subplots=3):
        """
        Plot input, label, and (optionally) model predictions for a given label column.
        By default, plots the first label_column if plot_col is not specified.
        """
        inputs, labels = self.sample_batch
        
        # Default to first label_column if plot_col not specified
        if plot_col is None:
            if self.label_columns:
                plot_col=self.label_columns[0]
            #pick the first column in the dataframe to avoid crash
            else:
                plot_col=list(self.column_indices.keys())[0]


        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs['numerical']))

        # Get the country code for each example in the batch (last timestep of the window)
        country_codes = inputs['country'][:, -1].numpy()  # shape: (batch,)

        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs['numerical'][n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)

            if model is not None:
                predictions = model(inputs)

                # Get prediction shape and slice label_indices accordingly
                pred_values = predictions[n, :, label_col_index]
                pred_len = pred_values.shape[0]
                sliced_indices = self.label_indices[:pred_len]

                plt.scatter(sliced_indices, pred_values,
                            marker='X', edgecolors='k', label='Predictions',
                            c='red', s=64)
            
            # Annotate with country code
            plt.title(f'CountryCode: {country_codes[n]}')

            if n == 0:
                plt.legend()

        plt.xlabel('Time (h)')
        plt.tight_layout()
        plt.show()
    
    def make_dataset_per_country(self, data):
        """
        Creates a combined `tf.data.Dataset` from per-country time series data,
        ensuring that sliding windows are generated independently for each country.

        This method:
        - Loops over all unique CountryCode values in the input DataFrame.
        - For each country, generates non-overlapping time windows using 
        tf.keras.preprocessing.timeseries_dataset_from_array.
        - Applies the `split_to_inputs_labels` mapping to separate inputs and labels.
        - Concatenates all per-country datasets into a single unified dataset.

        This prevents windows from crossing country boundaries, which would otherwise
        happen if the data were treated as a single long sequence.

        Parameters
        ----------
        data : pd.DataFrame
            The full dataset containing all countries. Must include the column 'CountryCode'.

        Returns
        -------
        tf.data.Dataset
            A concatenated dataset of input/label pairs, built separately for each country.
        """
        data = data.copy()
        country_datasets = []

        for code in sorted(data['CountryCode'].unique()):
            country_df = data[data['CountryCode'] == code]
            country_array = np.array(country_df, dtype=np.float32)

            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=country_array,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=self.shift,
                shuffle=self.shuffle,
                batch_size=32
            )

            ds = ds.map(self.split_to_inputs_labels)
            country_datasets.append(ds)

        # Combine all datasets safely
        full_ds = reduce(lambda ds1, ds2: ds1.concatenate(ds2), country_datasets)
        return full_ds


    def make_dataset(self, data):
        return self.make_dataset_per_country(data)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        "For plotting or inspection: retrieves and caches a sample batch from the training dataset."
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result


class RepeatBaseline(tf.keras.Model):
    "RepeatBaseline forecasts the next 24 hours using the last 24 observed hours"
    def __init__(self, label_index=None, input_width=24, label_width=24):
        super().__init__()
        self.label_index = label_index
        self.input_width = input_width
        self.label_width = label_width

    def call(self, inputs):
        # Take the last 24 steps from inputs['numerical']
        past_values = inputs['numerical'][:, -self.label_width:, self.label_index:self.label_index+1]  # shape: (batch, 24, 1)
        return past_values



def compile_fit(model,window,patience=3,max_epoch=50):
  """
  The function takes a a deep learning model and a window of data from the DataWindow class
  The patience is the number of epochs after which the model should stop training if the validation loss does not improve
  max_epochs sets a maximum number of epochs to train the model
  Validation dataset is used to calculate validation loss, where the loss is monitored 
  using EarlyStopping which will stop after 3 epochs in case the loss isn't reduced
  """
  #mode='min': Stops if no decrease is seen
  early_stopping=EarlyStopping(monitor='val_loss',patience=patience,mode='min')
  # The MSE is used as the loss function.
  # The MAE is used as an error metric.
  # Compiled in Keras is simply configures the model to specify the loss function to be used, the optimizer, and metrics of evaluation.
  # This uses the default learning rate for the Adam optimizer, which is 0.001 (1e-3) in Keras/TensorFlow.
  model.compile(loss=MeanSquaredError(),optimizer=Adam(),metrics=[MeanAbsoluteError()])
  # The model is fit on the training set.
  # early_stopping is passed as a callback to avoids overfitting.
  history=model.fit(window.train,epochs=max_epoch,validation_data=window.val,callbacks=[early_stopping],verbose=0)

  return history


class AutoRegressiveLSTM(Model):
    """
    An autoregressive LSTM model for multistep time series forecasting.

    This model:
    - Learns from a sequence of numerical features and a categorical 'CountryCode' using embedding.
    - Uses an LSTM cell to capture temporal patterns from the input sequence.
    - Predicts future timesteps recursively: the first prediction comes from the full input, 
    while the rest are predicted autoregressively using the previous output.

    Args:
        units (int): Number of hidden units in the LSTM cell.
        out_steps (int): Number of future timesteps to predict.
        num_numerical_features (int): Number of numerical input features (excluding categorical).
        embedding_dim (int): Size of the embedding vector for the country code (default=4).
        num_countries (int): Number of unique country codes (default=24).
    """
    def __init__(self, units, out_steps, num_numerical_features, embedding_output_dim=4, num_countries=24):
        super().__init__()
        self.out_steps = out_steps
        # The number of “hidden units” (neurons) in your LSTM cell. Which control:
        # The capacity of the LSTM to learn patterns.
        # More units = more “memory” and power to capture complex relationships, but also more risk of overfitting and higher computational cost.
        self.units = units

        # Country embedding
        # embedding_dim: The size of the dense vector used to represent each country (or other categorical variable). Which control:
        # How much information the model can store about each country.
        # Larger values can let the model learn more subtle differences, but too large may overfit or waste resources.
        self.country_embedding = layers.Embedding(input_dim=num_countries, output_dim=embedding_output_dim)

        # LSTM Cell + RNN wrapper
        self.lstm_cell = layers.LSTMCell(units)
        self.lstm_rnn = layers.RNN(self.lstm_cell, return_state=True)

        # Dense layer to predict 'Value' (1 target)
        self.dense = layers.Dense(1)

        # Track feature dimensions for concatenation
        self.num_numerical_features = num_numerical_features
        #wraps an LSTM cell to process sequences and return its internal memory state, 
        # allowing manual autoregressive forecasting across timesteps.
        self.embedding_output_dim = embedding_output_dim

    def warmup(self, numerical_inputs, country_inputs):
        # Embed country and concatenate to numerical
        country_emb = self.country_embedding(country_inputs)  # (batch, time, emb)
        x = layers.Concatenate(axis=-1)([numerical_inputs, country_emb])  # (batch, time, features + emb)

        # Run LSTM and get final state
        x, *state = self.lstm_rnn(x)
        prediction = self.dense(x)  # First prediction from last LSTM output

        return prediction, state

    def call(self, inputs, training=None):
        numerical_inputs = inputs['numerical']     # (batch, 24, features)
        country_inputs = inputs['country']         # (batch, 24)

        # Only need last timestep of country code for recursive prediction
        last_country = country_inputs[:, -1]       # (batch,)

        predictions = []
        prediction, state = self.warmup(numerical_inputs, country_inputs)
        predictions.append(prediction)

        for _ in range(1, self.out_steps):
            x_val = prediction
            dummy_numericals = tf.zeros_like(x_val)
            dummy_numericals = tf.repeat(dummy_numericals, repeats=self.num_numerical_features - 1, axis=-1)
            x_numerical = tf.concat([x_val, dummy_numericals], axis=-1)  # (batch, num_numerical_features)

            country_emb = self.country_embedding(last_country)  # (batch, emb)
            x = tf.concat([x_numerical, country_emb], axis=-1)  # (batch, num_features + emb)

            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        # Shape: (out_steps, batch, 1) -> (batch, out_steps, 1)
        return tf.transpose(tf.stack(predictions), [1, 0, 2])
    

class WaveNetBlock(tf.keras.layers.Layer):
    """
    A single dilated causal convolutional block used in WaveNet architecture.

    This block captures long-range temporal dependencies using dilated convolutions,
    and includes gated activation units, skip connections, and residual connections
    for efficient gradient flow and deep stacking.

    Parameters:
    -----------
    filters : int
        Number of filters (output channels) for the convolutions in the block.
    kernel_size : int
        Size of the convolutional kernel.
    dilation_rate : int
        Dilation rate for the causal convolutions, controlling the receptive field size.

    Attributes:
    -----------
    conv_filter : Conv1D
        Causal dilated convolution with tanh activation for the filter path.
    conv_gate : Conv1D
        Causal dilated convolution with sigmoid activation for the gating path.
    conv_skip : Conv1D
        1x1 convolution to produce skip connection output.
    conv_residual : Conv1D
        1x1 convolution to produce residual output.
    input_projection : Conv1D or None
        Optional 1x1 convolution to align input dimensions for residual addition.

    Methods:
    --------
    call(x):
        Performs gated activation, computes skip and residual outputs.
        Returns:
            skip: Tensor of shape (batch, time, filters)
            residual_output: Tensor added to input for residual learning
    """
    def __init__(self, filters, kernel_size, dilation_rate):
        super().__init__()
        self.filters = filters
        # tanh and sigmoid allows the model to learn both content and flow control, which is powerful for time series.
        self.conv_filter = layers.Conv1D(filters, kernel_size, padding='causal',
                                  dilation_rate=dilation_rate, activation='tanh')
        self.conv_gate = layers.Conv1D(filters, kernel_size, padding='causal',
                                dilation_rate=dilation_rate, activation='sigmoid')
        self.conv_skip = layers.Conv1D(filters, 1)
        self.conv_residual = layers.Conv1D(filters, 1)

        self.input_projection = None  # will be created dynamically in build()

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.filters:
            self.input_projection = layers.Conv1D(self.filters, 1)  # Project to match residual
        super().build(input_shape)

    def call(self, x):
        if self.input_projection is not None:
            x_proj = self.input_projection(x)
        else:
            x_proj = x

        z_filter = self.conv_filter(x)
        z_gate = self.conv_gate(x)
        z = z_filter * z_gate

        skip = self.conv_skip(z)
        residual = self.conv_residual(z)
        return skip, x_proj + residual

class WaveNetModel(Model):
    """
    WaveNet-style autoregressive model for multi-step time series forecasting.

    This model combines dilated causal convolutions with country embeddings
    to capture both local temporal patterns and country-specific characteristics.

    Args:
        out_steps (int): Number of future time steps to predict.
        num_numerical_features (int): Number of continuous input features.
        num_countries (int): Number of unique country codes (used for embedding).
        embedding_dim (int): Dimension of the country embedding vector.
        filters (int): Number of filters in each convolution layer.
        kernel_size (int): Size of the 1D convolution kernel.
        num_blocks (int): Number of dilated convolution blocks (each with increasing dilation).

    Architecture:
        - Embeds categorical country codes.
        - Concatenates embeddings with numerical features.
        - Passes through a stack of dilated convolutional blocks.
        - Applies final convolution and dense layer to produce multi-step forecasts.

    Inputs:
        inputs (dict): {
            'numerical': Tensor of shape (batch_size, time_steps, num_numerical_features),
            'country': Tensor of shape (batch_size, time_steps)
        }

    Output:
        Tensor of shape (batch_size, time_steps, out_steps), containing predictions for each time step.
    """
    def __init__(self, out_steps, num_numerical_features, num_countries, embedding_output_dim,
                 filters=32, kernel_size=2, num_blocks=4):
        super().__init__()
        self.out_steps = out_steps

        self.embedding = layers.Embedding(input_dim=num_countries, output_dim=embedding_output_dim)

        #In time series forecasting, using dilation_rate > 1 allows the model to look further into the past 
        # without increasing the kernel size or reducing resolution.
        self.blocks = [
            WaveNetBlock(filters, kernel_size, dilation_rate=2 ** i)
            for i in range(num_blocks)
        ]

        self.final_conv = layers.Conv1D(filters, 1, activation='relu')
        self.output_layer = layers.Dense(out_steps)

    def call(self, inputs):
        x_num = inputs['numerical']  # (batch, time, features)
        x_country = inputs['country']  # (batch, time)
        x_emb = self.embedding(x_country)  # (batch, time, embedding_dim)

        x = tf.concat([x_num, x_emb], axis=-1)  # (batch, time, total_features)

        skip_connections = []

        for block in self.blocks:
            skip, x = block(x)
            skip_connections.append(skip)

        x = tf.add_n(skip_connections)
        x = self.final_conv(x)
        x = self.output_layer(x)  # (batch, time, out_steps)

        return x

class ModelBuilder:
    """
    A factory class for building different neural network models for multivariate time series forecasting.

    This class simplifies:
    - Creating input layers for numerical and categorical (embedded) features.
    - Building various model architectures, including baseline, linear, dense, CNN, LSTM, hybrid CNN+LSTM,
      autoregressive LSTM, and WaveNet.
    - Managing consistent input shapes and embeddings based on a shared DataWindow configuration.

    Parameters
    ----------
    window : DataWindow
        The DataWindow object defining input/output dimensions and feature alignment.
    train : pd.DataFrame
        Training DataFrame used to extract feature metadata.

    Methods
    -------
    _make_inputs(drop_column, embedding_input_dim, embedding_output_dim)
        Internal method to create shared model input layers and country embedding.
    
    repeat_baseline(target, input_width, label_width)
        Builds a baseline model that repeats the last input value across the prediction window.
    
    linear(drop_column, embedding_input_dim, embedding_output_dim)
        Builds a simple linear model that maps concatenated inputs to output directly using a Dense layer.
    
    fnn(drop_column, embedding_input_dim, embedding_output_dim)
        Builds a feedforward (dense) neural network applied in a time-distributed manner.
    
    lstm(drop_column, embedding_input_dim, embedding_output_dim, lstm_units)
        Builds an LSTM model for sequential modeling with embedded country input.
    
    cnn(drop_column, embedding_input_dim, embedding_output_dim, filters)
        Builds a 1D Convolutional model to capture local temporal patterns.
    
    cnn_lstm(drop_column, embedding_input_dim, embedding_output_dim, filters, lstm_units)
        Builds a hybrid model combining CNN and LSTM layers.
    
    arlstm(units, out_steps, drop_column, embedding_dim, num_countries)
        Builds an AutoRegressive LSTM model that recursively predicts each step using previous outputs.
    
    wavenet(drop_column, out_steps, embedding_dim, num_countries, filters, kernel_size)
        Builds a WaveNet-style model using dilated causal convolutions for deep temporal receptive fields.
    """

    def __init__(self,window,train):
        self.window=window
        self.train=train
        self.column_indices = {name: i for i, name in enumerate(self.train.columns)}


    def _make_inputs(self,drop_column:list,embedding_input_dim=24,embedding_output_dim=4):
        # Create Keras input and embedding layers for models
        num_numerical_features = self.train.drop(columns=drop_column).shape[1]
        #Input: is a Keras layer that defines a model input. It tells the model what kind of data it should expect.
        numerical_input = layers.Input(shape=(self.window.input_width, num_numerical_features), name='numerical')
        country_input = layers.Input(shape=(self.window.input_width,), dtype=tf.int32, name='country')
        country_emb = layers.Embedding(input_dim=embedding_input_dim, output_dim=embedding_output_dim)(country_input)
        return num_numerical_features,numerical_input,country_input,country_emb

    def repeat_baseline(self,target='Value', input_width=24, label_width=24):
        label_index=self.column_indices[target]
        return RepeatBaseline(label_index=label_index, input_width=input_width, label_width=label_width)

    def linear(self,drop_column:list, embedding_input_dim=24, embedding_output_dim=4):
        _,numerical_input,country_input,country_emb=self._make_inputs(drop_column, embedding_input_dim, embedding_output_dim)
        x = layers.Concatenate(axis=-1)([numerical_input, country_emb])
        output = layers.Dense(1, kernel_initializer=tf.initializers.zeros())(x)
        model = Model(inputs={'numerical': numerical_input, 'country': country_input}, outputs=output)
        return model

    def fnn(self,drop_column:list, embedding_input_dim=24, embedding_output_dim=4):
        _,numerical_input,country_input,country_emb=self._make_inputs(drop_column, embedding_input_dim, embedding_output_dim)
        x = layers.Concatenate(axis=-1)([numerical_input, country_emb])
        # 'relu'nonlinear activation function to capture nonlinear relationships in the data
        # Without TimeDistributed, a Dense layer flattens the sequence, breaking the time structure.
        x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
        x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
        output = layers.TimeDistributed(layers.Dense(1, kernel_initializer=tf.initializers.zeros()))(x)
        model = Model(inputs={'numerical': numerical_input, 'country': country_input}, outputs=output)
        return model

    def lstm(self,drop_column:list, embedding_input_dim=24, embedding_output_dim=4, lstm_units=32):
        """lstm_units: specifies how many hidden units (or memory cells) the LSTM layer uses
            embedding_output_dim=4 means each country code is represented by a trainable 4-dimensional 
            vector capturing country-specific patterns."""
        _,numerical_input,country_input,country_emb=self._make_inputs(drop_column, embedding_input_dim, embedding_output_dim)
        x = layers.Concatenate(axis=-1)([numerical_input, country_emb])
        # return_sequences=True, so you don’t need to wrap it in TimeDistributed.
        x = layers.LSTM(lstm_units, return_sequences=True)(x)
        output = layers.TimeDistributed(layers.Dense(1, kernel_initializer=tf.initializers.zeros()))(x)
        model = Model(inputs={'numerical': numerical_input, 'country': country_input}, outputs=output)
        return model

    def cnn(self,drop_column:list, embedding_input_dim=24, embedding_output_dim=4, filters=32,kernel_size=3):
        """filters/kernels: The number of learnable patterns the network uses to scan over the input sequence or image."""
        _,numerical_input,country_input,country_emb=self._make_inputs(drop_column, embedding_input_dim, embedding_output_dim)
        x = layers.Concatenate(axis=-1)([numerical_input, country_emb])
        #kernel_size=3 This sets the width of the sliding window (the convolution filter) to 3 timesteps.
        # It slides a convolutional kernel over time steps, so you do NOT need TimeDistributed.
        #padding='causal' Ensures that the output at time t only depends on inputs at or before time t — not from the future
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='causal')(x)
        x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
        output = layers.TimeDistributed(layers.Dense(1, kernel_initializer=tf.initializers.zeros()))(x)
        model = Model(inputs={'numerical': numerical_input, 'country': country_input}, outputs=output)
        return model

    def cnn_lstm(self,drop_column:list, embedding_input_dim=24, embedding_output_dim=4, filters=32, lstm_units=32,kernel_size=3):
        _,numerical_input,country_input,country_emb=self._make_inputs(drop_column, embedding_input_dim, embedding_output_dim)
        x = layers.Concatenate(axis=-1)([numerical_input, country_emb])
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='causal')(x)
        x = layers.LSTM(lstm_units, return_sequences=True)(x)
        output = layers.TimeDistributed(layers.Dense(1, kernel_initializer=tf.initializers.zeros()))(x)
        model = Model(inputs={'numerical': numerical_input, 'country': country_input}, outputs=output)
        return model

    
    def arlstm(self, units, out_steps, drop_column:list, embedding_output_dim, num_countries):
        num_numerical_features = self.train.drop(columns=drop_column).shape[1]
        return AutoRegressiveLSTM(
            units=units,
            out_steps=out_steps,
            num_numerical_features=num_numerical_features,
            embedding_output_dim=embedding_output_dim,
            num_countries=num_countries
        )

    def wavenet(self,drop_column:list,out_steps,embedding_output_dim,num_countries,filters,kernel_size):
        num_numerical_features = self.train.drop(columns=drop_column).shape[1]
        return WaveNetModel(
            out_steps=out_steps,
            num_numerical_features=num_numerical_features,
            num_countries=num_countries,
            embedding_output_dim=embedding_output_dim,
            filters=filters,
            kernel_size=kernel_size)




def get_memory_mb():
    """Returns memory usage (MB) for current process."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def pipeline(model_type='lstm', model_kwargs=None, window=None,builder=None,val_perf=None,test_perf=None):
    """
    Build, train, and evaluate a model on a given DataWindow, tracking memory usage.
    
    Args:
        model_type (str): Model to build ('lstm', 'cnn', etc.).
        model_kwargs (dict): Arguments for the model.
        window (DataWindow): DataWindow object providing datasets.
        builder (object): ModelBuilder object with a method for each model_type.
        val_perf (dict): Dictionary to store validation scores.
        test_perf (dict): Dictionary to store test scores.

    Returns:
        model, history, val_perf, test_perf
    """
    # 1. Data prep
    if window is None:
        raise ValueError("You must provide a DataWindow instance to the pipeline.")


    # 2. Model build
    if builder is None:
        raise ValueError("You must provide a ModelBuilder instance with window and train.")
    
    if model_kwargs is None:
        model_kwargs = {}
    model_func = getattr(builder, model_type)
    model = model_func(**model_kwargs)

    # Calc memory before fitting
    mem_before = get_memory_mb()

    # 3. Training
    history = compile_fit(model, window)

    # 4. Evaluation
    val_perf[model_type] = model.evaluate(window.val)
    test_perf[model_type] = model.evaluate(window.test,verbose=0)

    # Calc memory after evaluation
    mem_after = get_memory_mb()

    print(f"Memory used for training: {mem_after - mem_before:.1f} MB")
    
    return model, history, val_perf, test_perf



def build_model_factory(builder, model_input):
    """
    Creates a Keras model-building function for use with hyperparameter tuning.

    This factory function returns a model-building function (`build_model(hp)`) that 
    selects model architecture and configuration based on the `model_input` type 
    (e.g., 'cnn_lstm' or 'wavenet') and hyperparameter choices provided by Keras Tuner.

    Parameters:
    -----------
    builder : object
        A model builder class or module with methods like `cnn_lstm` and `wavenet` that return compiled models.
    model_input : str
        A string specifying which model architecture to build ('cnn_lstm' or 'wavenet').

    Returns:
    --------
    function
        A function `build_model(hp)` compatible with Keras Tuner, which builds and compiles a model
        using the specified hyperparameters.
    """
    def build_model(hp):
        filters = hp.Choice('filters', [8, 16, 32, 64])
        kernel_size = hp.Choice('kernel_size', [2, 3, 5])
        embedding_output_dim = hp.Choice('embedding_output_dim', [2, 4, 8 ,16])
        learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 5e-4])
        lstm_units = hp.Choice('lstm_units', [16, 32, 64]) if model_input == 'cnn_lstm' else None

        model_fn = getattr(builder, model_input)

        if model_input == 'cnn_lstm':
            model = model_fn(
                drop_column=['CountryCode'],
                embedding_input_dim=24,
                embedding_output_dim=embedding_output_dim,
                filters=filters,
                kernel_size=kernel_size,
                lstm_units=lstm_units
            )
        elif model_input == 'wavenet':
            model = model_fn(
                drop_column=['CountryCode'],
                out_steps=24,
                embedding_output_dim=embedding_output_dim,
                num_countries=24,
                filters=filters,
                kernel_size=kernel_size
            )
        else:
            raise ValueError(f"Unsupported model_input: {model_input}")

        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['mae']
        )
        return model

    return build_model


def tuning(seed_list: list, builder, model_input: str, window):
    """
    Perform hyperparameter tuning for a given model using multiple seeds and Keras Tuner (RandomSearch).

    This function tunes either a `cnn_lstm` or `wavenet` model across multiple random seeds,
    averages the performance, and returns the best-performing model based on validation MAE.

    Parameters:
    -----------
    seed_list : list
        A list of random seeds for repeated tuning to account for variance in training.
    builder : object
        A model builder class or module that provides methods like `cnn_lstm` or `wavenet` 
        to construct models from hyperparameters.
    model_input : str
        Model type to tune; must match one of the builder's method names (e.g., 'cnn_lstm', 'wavenet').
    window : object
        A custom object with `.train`, `.val`, and `.test` datasets (e.g., a TimeSeriesWindow object).

    Returns:
    --------
    tuple
        - best_model (tf.keras.Model): The model with the best validation MAE across all seeds.
        - best_hyperparameters (dict): Dictionary of the best hyperparameter values.
        - weights_path (str): Path to the checkpoint weights file of the best model.

    Notes:
    ------
    - Uses early stopping with patience=2.
    - Limits each tuner to `max_trials=5`.
    - Assumes builder models accept specific keyword arguments (e.g., `filters`, `embedding_output_dim`).
    - Automatically loads the best weights from disk after tuning.
    """
    results = []

    build_model = build_model_factory(builder, model_input)

    for seed in seed_list:
        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.clear_session()

        tuner = RandomSearch(
            build_model,
            objective='val_mae',
            max_trials=5,
            executions_per_trial=1,
            directory='tuning_est',
            project_name=f'{model_input}_tuning_seed_{seed}'
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )

        tuner.search(
            window.train,
            validation_data=window.val,
            epochs=10,
            callbacks=[early_stop],
            verbose=2
        )

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        model_fn = getattr(builder, model_input)

        # Rebuild best model
        if model_input == 'cnn_lstm':
            best_model = model_fn(
                drop_column=['CountryCode'],
                embedding_input_dim=24,
                embedding_output_dim=best_hp.get('embedding_output_dim'),
                filters=best_hp.get('filters'),
                kernel_size=best_hp.get('kernel_size'),
                lstm_units=best_hp.get('lstm_units')
            )
        elif model_input == 'wavenet':
            best_model = model_fn(
                drop_column=['CountryCode'],
                out_steps=24,
                embedding_output_dim=best_hp.get('embedding_output_dim'),
                num_countries=24,
                filters=best_hp.get('filters'),
                kernel_size=best_hp.get('kernel_size')
            )

        best_model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=best_hp.get('learning_rate')),
            metrics=['mae']
        )

        sample_inputs, _ = next(iter(window.train))
        _ = best_model(sample_inputs)

        best_trial = tuner.oracle.get_best_trials(1)[0]
        trial_id = best_trial.trial_id
        if not trial_id.startswith('trial_'):
            trial_id = f"trial_{trial_id}"
        trial_folder = os.path.join('tuning_est', f'{model_input}_tuning_seed_{seed}', trial_id)
        weights_path = os.path.join(trial_folder, "checkpoint.weights.h5")
        best_model.load_weights(weights_path)

        test_eval = best_model.evaluate(window.test, verbose=0)
        results.append({
            'seed': seed,
            'val_mae': best_trial.metrics.get_last_value('val_mae'),
            'test_mae': test_eval[1],
            'hyperparameters': best_hp.values,
            'weights_path': weights_path,
            'model': best_model
        })

    best_result = min(results, key=lambda x: x['val_mae'])

    print(f"\n===== Best {model_input} Result Across Seeds =====")
    print("Best Seed:", best_result['seed'])
    print("Best Validation MAE:", best_result['val_mae'])
    print("Best Test MAE:", best_result['test_mae'])
    print("Best Hyperparameters:", best_result['hyperparameters'])
    print("Weights loaded from:", best_result['weights_path'])

    return best_result['model'], best_result['hyperparameters'], best_result['weights_path']




