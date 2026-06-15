from __future__ import annotations


def _transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    import tensorflow as tf

    attention = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention = tf.keras.layers.Dropout(dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    feed_forward = tf.keras.layers.Dense(ff_dim, activation="relu")(attention)
    feed_forward = tf.keras.layers.Dropout(dropout)(feed_forward)
    feed_forward = tf.keras.layers.Dense(inputs.shape[-1])(feed_forward)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention + feed_forward
    )


def build_rads_forecaster(
    lookback: int,
    numerical_dim: int,
    text_dim: int,
    *,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 64,
    lstm_units: int = 64,
    dropout: float = 0.20,
    use_text_transformer: bool = True,
    use_cross_attention: bool = True,
    use_rads: bool = True,
):
    """Build the full model or one of the manuscript's ablation variants."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Forecast model requires the optional 'deep' dependencies.") from exc

    numerical_input = tf.keras.Input(
        shape=(lookback, numerical_dim), name="numerical_sequence"
    )
    text_input = tf.keras.Input(shape=(lookback, text_dim), name="text_sequence")
    rads_input = tf.keras.Input(shape=(1,), name="rads_score")

    numerical_encoded = _transformer_encoder(
        numerical_input, head_size, num_heads, ff_dim, dropout
    )
    numerical_encoded = tf.keras.layers.LSTM(
        lstm_units, return_sequences=True, name="numerical_lstm"
    )(numerical_encoded)
    text_encoded = (
        _transformer_encoder(text_input, head_size, num_heads, ff_dim, dropout)
        if use_text_transformer
        else text_input
    )
    text_encoded = tf.keras.layers.LSTM(
        lstm_units, return_sequences=True, name="text_lstm"
    )(text_encoded)

    if use_cross_attention:
        text_to_market = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, name="text_queries_market"
        )(text_encoded, numerical_encoded)
        market_to_text = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, name="market_queries_text"
        )(numerical_encoded, text_encoded)
        fused = tf.keras.layers.Concatenate(name="dual_cross_attention")(
            [text_to_market, market_to_text]
        )
        fused = tf.keras.layers.GlobalAveragePooling1D()(fused)
    else:
        numerical_vector = tf.keras.layers.GlobalAveragePooling1D()(
            numerical_encoded
        )
        text_vector = tf.keras.layers.GlobalAveragePooling1D()(text_encoded)
        fused = tf.keras.layers.Concatenate(name="plain_multimodal_fusion")(
            [numerical_vector, text_vector]
        )
    augmented = (
        tf.keras.layers.Concatenate(name="fusion_plus_rads")([fused, rads_input])
        if use_rads
        else fused
    )
    augmented = tf.keras.layers.Dense(64, activation="relu")(augmented)
    augmented = tf.keras.layers.Dropout(dropout)(augmented)
    output = tf.keras.layers.Dense(1, name="next_price")(augmented)
    inputs = (
        [numerical_input, text_input, rads_input]
        if use_rads
        else [numerical_input, text_input]
    )
    model = tf.keras.Model(
        inputs,
        output,
        name="rads_multimodal_forecaster",
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
