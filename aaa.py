from mango.data import get_ts_dataset
import random
import numpy as np
from mango.processing import create_recurrent_dataset

random.seed(42)
df = get_ts_dataset()
df["new_feature"] = [random.randrange(1, 5) for _ in range(len(df))]
df["other_new_feature"] = [random.randrange(10, 50) for _ in range(len(df))]
df = df[["new_feature", "other_new_feature", "date", "target"]]
print(df)
data = df.to_numpy()

include_output_lags = True
look_back = 4
output_last = True


x, y = create_recurrent_dataset(
    data, look_back, include_output_lags, lags=[1, 7], output_last=output_last
)
print(x.shape, y.shape)
print(x)
print(y)
