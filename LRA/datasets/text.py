import pickle
import numpy as np
import input_pipeline_tc
import sys
sys.path.append("./long-range-arena/lra_benchmarks/text_classification/")

train_ds, eval_ds, test_ds, encoder = input_pipeline_tc.get_tc_datasets(
    n_devices=1, task_name="imdb_reviews", data_dir=None,
    batch_size=1, fixed_vocab=None, max_length=1000)

mapping = {"train": train_ds, "dev": eval_ds, "test": test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0": np.concatenate([inst["inputs"].numpy()[0], np.zeros(24, dtype=np.int32)]),
            "label": inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end="\r")
    with open(f"text.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
