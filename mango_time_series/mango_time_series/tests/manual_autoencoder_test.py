from mango_time_series.models.autoencoder import AutoEncoder
from mango.data import load_energy_prices_dataset

TIME_STEPS = 24
SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 256
TIME_STEP_TO_CHECK = -1


def train_autoencoder():
    data = load_energy_prices_dataset(
        add_features=["hour", "month", "dayofweek"],
        output_format="numpy",
        dummy_features=True,
    )

    model = AutoEncoder(
        form="gru",
        data=data,
        context_window=TIME_STEPS,
        time_step_to_check=TIME_STEP_TO_CHECK,
        num_layers=3,
        hidden_dim=[15, 10, 5],
        normalize=True,
        batch_size=BATCH_SIZE,
        split_size=SPLIT,
        epochs=EPOCHS,
        save_path=f"../data",
        verbose=True,
    )
    model.train()

    # _ = model.reconstruct()


if __name__ == "__main__":
    train_autoencoder()
