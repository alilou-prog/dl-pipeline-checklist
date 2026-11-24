"""Configurable Keras MLP builder and reproducible experiment runner."""

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .evaluation import evaluate_predictions


def build_nn(
    input_dim: int,
    hidden_units=(128, 64, 32),
    activation: str = "elu",
    initializer: str = "he_normal",
    l2_reg: float = 0.0,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = False,
):
    """Builds a feed-forward binary classifier with configurable depth, activation, init, BatchNorm, and regularization."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for units in hidden_units:
        if use_batchnorm:
            model.add(
                layers.Dense(
                    units,
                    use_bias=False,
                    kernel_initializer=initializer,
                    kernel_regularizer=reg,
                )
            )
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation))
        else:
            model.add(
                layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer=initializer,
                    kernel_regularizer=reg,
                )
            )
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer=initializer))
    return model


def run_nn_experiment(
    name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_units=(128, 64, 32),
    activation: str = "elu",
    initializer: str = "he_normal",
    optimizer_fn=lambda: keras.optimizers.Adam(learning_rate=1e-3),
    l2_reg: float = 0.0,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = False,
    loss="binary_crossentropy",
    epochs: int = 50,
    batch_size: int = 256,
    callbacks=None,
    class_weight=None,
    seed: int = 42,
):
    """Trains a single NN configuration and evaluates both train and validation splits to check overfitting."""
    keras.utils.set_random_seed(seed)
    model = build_nn(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        activation=activation,
        initializer=initializer,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
    )
    model.compile(
        optimizer=optimizer_fn(),
        loss=loss,
        metrics=[
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks or [],
        class_weight=class_weight,
        verbose=0,
    )
    train_prob = model.predict(X_train, batch_size=1024, verbose=0).ravel()
    val_prob = model.predict(X_val, batch_size=1024, verbose=0).ravel()
    epochs_ran = len(history.history["loss"])

    train_metrics = evaluate_predictions(name, y_train, (train_prob >= 0.5).astype(int), train_prob, split="Train")
    val_metrics = evaluate_predictions(name, y_val, (val_prob >= 0.5).astype(int), val_prob, split="Val")
    train_metrics["Epochs Ran"] = epochs_ran
    val_metrics["Epochs Ran"] = epochs_ran
    return model, history, train_metrics, val_metrics, train_prob, val_prob
