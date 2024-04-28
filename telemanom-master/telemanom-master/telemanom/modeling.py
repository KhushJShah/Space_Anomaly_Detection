import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tcn import TCN  # Import TCN layer

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')

class Model:
    def __init__(self, config, run_id, channel):
        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None

        if not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                path = os.path.join('data', self.config.use_id, 'models', self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing model at {}'.format(path))
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()

    def load(self):
        logger.info('Loading pre-trained model')
        self.model = load_model(os.path.join('data', self.config.use_id, 'models', self.chan_id + '.h5'))

    def train_new(self, channel):
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.config.patience, min_delta=self.config.min_delta, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
        checkpoint = ModelCheckpoint(os.path.join('data', self.run_id, 'models', '{}.h5'.format(self.chan_id)), save_best_only=True, monitor='val_loss', mode='min')

        self.model = Sequential([
            TCN(input_shape=(None, channel.X_train.shape[2]), nb_filters=64, kernel_size=6, dilations=[1, 2, 4, 8], padding='causal'),
            Dropout(self.config.dropout),
            Dense(self.config.n_predictions, activation='linear', kernel_regularizer=l2(0.01))
        ])

        # Use Adam optimizer with a dynamic learning rate
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss=self.config.loss_metric, optimizer=optimizer)
        self.model.fit(channel.X_train, channel.y_train, batch_size=self.config.lstm_batch_size, epochs=self.config.epochs, validation_split=self.config.validation_split, callbacks=[early_stopping, reduce_lr, checkpoint], verbose=1)

    def save(self):
        self.model.save(os.path.join('data', self.run_id, 'models', '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='first'):
        agg_y_hat_batch = np.array([])
        for t in range(len(y_hat_batch)):
            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()
            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))
        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        num_batches = int((channel.y_test.shape[0] - self.config.l_s) / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'.format(self.config.l_s, channel.y_test.shape[0]))
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size
            if i + 1 == num_batches + 1:
                idx = channel.y_test.shape[0]
            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.model.predict(X_test_batch)
            self.aggregate_predictions(y_hat_batch)
        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))
        channel.y_hat = self.y_hat
        np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'.format(self.chan_id)), self.y_hat)

        return channel
