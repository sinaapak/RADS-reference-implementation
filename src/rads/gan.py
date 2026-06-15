from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class HeadlineVocabulary:
    tokens: list[str]
    max_length: int

    @classmethod
    def fit(
        cls, headlines: Sequence[str], max_length: int = 16, max_vocabulary: int = 2000
    ) -> "HeadlineVocabulary":
        counts = Counter(token for text in headlines for token in str(text).split())
        tokens = ["<pad>", "<unk>"] + [
            token for token, _ in counts.most_common(max_vocabulary - 2)
        ]
        return cls(tokens=tokens, max_length=max_length)

    def encode(self, headlines: Sequence[str]) -> np.ndarray:
        lookup = {token: index for index, token in enumerate(self.tokens)}
        encoded = np.zeros(
            (len(headlines), self.max_length, len(self.tokens)), dtype=np.float32
        )
        for row, headline in enumerate(headlines):
            words = str(headline).split()[: self.max_length]
            words += ["<pad>"] * (self.max_length - len(words))
            for column, token in enumerate(words):
                encoded[row, column, lookup.get(token, 1)] = 1.0
        return encoded

    def decode(self, distributions: np.ndarray) -> list[str]:
        ids = np.asarray(distributions).argmax(axis=-1)
        return [
            " ".join(
                self.tokens[index]
                for index in row
                if self.tokens[index] not in {"<pad>", "<unk>"}
            )
            for row in ids
        ]


def build_dense_headline_gan(
    vocabulary_size: int,
    max_length: int,
    latent_dim: int = 100,
    learning_rate: float = 0.0002,
):
    """Table 2 dense GAN with two 128-unit LeakyReLU generator layers."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Headline GAN requires optional 'deep' dependencies.") from exc

    generator = tf.keras.Sequential(
        [
            tf.keras.layers.Input((latent_dim,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dense(max_length * vocabulary_size),
            tf.keras.layers.Reshape((max_length, vocabulary_size)),
            tf.keras.layers.Softmax(axis=-1),
        ],
        name="headline_generator",
    )
    discriminator = tf.keras.Sequential(
        [
            tf.keras.layers.Input((max_length, vocabulary_size)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="headline_discriminator",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    discriminator.compile(optimizer=optimizer, loss="binary_crossentropy")
    discriminator.trainable = False
    noise = tf.keras.Input((latent_dim,))
    validity = discriminator(generator(noise))
    combined = tf.keras.Model(noise, validity, name="headline_gan")
    combined.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )
    discriminator.trainable = True
    return generator, discriminator, combined


def generate_headlines(generator, vocabulary: HeadlineVocabulary, count: int) -> list[str]:
    latent_dim = generator.input_shape[-1]
    noise = np.random.normal(size=(count, latent_dim)).astype(np.float32)
    return vocabulary.decode(generator.predict(noise, verbose=0))


def train_dense_headline_gan(
    real_headlines: Sequence[str],
    *,
    epochs: int = 200,
    batch_size: int = 64,
    latent_dim: int = 100,
    max_length: int = 16,
    max_vocabulary: int = 2000,
    random_state: int = 42,
):
    """Train the Table 2 GAN and return it with its fitted vocabulary."""
    rng = np.random.default_rng(random_state)
    vocabulary = HeadlineVocabulary.fit(
        real_headlines,
        max_length=max_length,
        max_vocabulary=max_vocabulary,
    )
    real = vocabulary.encode(real_headlines)
    generator, discriminator, combined = build_dense_headline_gan(
        len(vocabulary.tokens),
        vocabulary.max_length,
        latent_dim=latent_dim,
    )
    half_batch = max(1, batch_size // 2)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        real_indices = rng.integers(0, len(real), size=half_batch)
        real_batch = real[real_indices]
        noise = rng.normal(size=(half_batch, latent_dim)).astype(np.float32)
        fake_batch = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        d_real = discriminator.train_on_batch(
            real_batch, np.ones((half_batch, 1), dtype=np.float32)
        )
        d_fake = discriminator.train_on_batch(
            fake_batch, np.zeros((half_batch, 1), dtype=np.float32)
        )
        discriminator.trainable = False
        noise = rng.normal(size=(batch_size, latent_dim)).astype(np.float32)
        g_loss = combined.train_on_batch(
            noise, np.ones((batch_size, 1), dtype=np.float32)
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "discriminator_loss": float((float(d_real) + float(d_fake)) / 2),
                "generator_loss": float(g_loss),
            }
        )
    discriminator.trainable = True
    return generator, discriminator, vocabulary, history
