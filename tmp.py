from pathlib import Path

from aim import Repo
from aim.storage.context import Context
from tqdm import tqdm

from stracker.ana import get_metric_names_of_aim_run, get_metric_value_of_aim_run

aim_path = Path("/Users/ei/Documents/aim/crowdsourcing")
repo = Repo(str(aim_path))
context = Context({})

removing_hashes = []
for run in tqdm(list(repo.iter_runs())):
    metrics = get_metric_names_of_aim_run(run)
    value = run.get_metric("test_batch_acc", context=context)
    # assert "test_batch_acc" in metrics, metrics
    # assert get_metric_value_of_aim_run(run) is not None
    if "test_batch_acc" not in metrics or value is None:
        # print(metrics)
        # print(run.hash)
        infos = list(run.iter_metrics_info())
        vs = [run.get_metric(info[0], context=info[1]) for info in infos]
        allnone = all(v is None for v in vs)
        print(allnone)
        # for m in metrics:
        #     v = run.get_metric(m, context=context)
        #     print(v is None)
        if allnone:
            removing_hashes.append(run.hash)

print(len(removing_hashes))
for h in removing_hashes:
    repo.delete_run(h)
