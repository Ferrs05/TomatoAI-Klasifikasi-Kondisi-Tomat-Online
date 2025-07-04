import os
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from sklearn.metrics import confusion_matrix
from data_prep import load_data_cnn, IMAGE_SIZE
from utils import print_metrics, plot_confusion_matrix, plot_training_history


LABELS = ['matang', 'belum_matang', 'rusak']


def build_model(hp):
    model = tf.keras.Sequential()
    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(*IMAGE_SIZE, 3)))

    # Conv-pool layers
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        filters = hp.Choice(f'filters_{i}', [32, 64, 128])
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flatten and dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            hp.Int('dense_units', 64, 256, step=64),
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.Dropout(
            hp.Float('dropout', 0.3, 0.7, step=0.2)
        )
    )
    model.add(tf.keras.layers.Dense(len(LABELS), activation='softmax'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_and_train(
    train_dir,
    test_dir,
    model_dir='models',
    max_trials=5,
    epochs=20
):
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    X_train, y_train = load_data_cnn(train_dir)
    X_test, y_test = load_data_cnn(test_dir)

    if X_train.size == 0 or y_train.size == 0:
        raise ValueError(
            f"Tidak ada data latih di '{train_dir}'. Pastikan folder dan file benar."
        )

    # RandomSearch tuner
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='tomat_rs',
        overwrite=True
    )
 # Early stopping
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    )

    # Search hyperparameters
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[stop],
        verbose=1
    )

    # Build and train final model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hp)

    ckpt_path = os.path.join(model_dir, 'best_tomat_cnn.keras')
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_best_only=True,
        monitor='val_accuracy'
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[ckpt, stop],
        verbose=1
    )
    
    # Evaluation
    if X_test.size == 0:
        print("Warning: Tidak ada data uji. Lewati evaluasi.")
    else:
        preds = model.predict(X_test, batch_size=32, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Metrics and plots
        print_metrics(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, normalize=True)
        plot_training_history(history)

    print(f"Model CNN terbaik disimpan di {ckpt_path}")