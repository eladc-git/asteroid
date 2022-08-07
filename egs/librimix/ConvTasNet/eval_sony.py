import os
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from models.convtasnetq import ConvTasNetQ
from models.load_model import enable_observer

from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.utils import tensors_to_device

parser = argparse.ArgumentParser()

parser.add_argument(
    "--exp_dir", type=str, required=True, help="Experiment root"
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory in exp_dir where the eval results" " will be stored",
)

parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)

parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)

parser.add_argument(
    "--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model"
)

COMPUTE_METRICS = ["si_sdr", "stoi"]

ASR_MODEL_PATH = (
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")

    # ------ QAT model --------------
    qat = conf["train_conf"]["training"]["qat"]
    model = ConvTasNetQ(n_src=conf["train_conf"]["data"]["n_src"], mask_act=conf["train_conf"]["masknet"]["mask_act"], qat=qat, unsqueeze_input=True)
    model.quantize_model()
    model_state_dict_weights = torch.load(model_path)
    model.load_state_dict(model_state_dict_weights, strict=True)
    enable_observer(model, False)
    # -------------------------------

    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    test_set = LibriMix(
        csv_dir=conf["train_conf"]["data"]["valid_dir"],
        task=conf["train_conf"]["data"]["task"],
        sample_rate=conf["train_conf"]["data"]["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        # Model
        est_sources = model(mix.unsqueeze(0))
        # Align: pad output to match clean waveform size
        pad_size = sources.shape[1] - est_sources.shape[2]
        est_sources = torch.nn.functional.pad(est_sources[0], [0, pad_size])
        # Loss
        loss, reordered_sources = loss_func(est_sources[None], sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        series_list.append(pd.Series(utt_metrics))

        # Print and save summary metrics
        if idx % 500 == 0 and idx > 0:
            final_results = {}
            all_metrics_df = pd.DataFrame(series_list)
            for metric_name in COMPUTE_METRICS:
                input_metric_name = "input_" + metric_name
                ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
                final_results[metric_name] = all_metrics_df[metric_name].mean()
                final_results[metric_name + "_imp"] = ldf.mean()
            print("Meanwhile metrics:")
            pprint(final_results)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in COMPUTE_METRICS:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()
    print("Overall metrics:")
    pprint(final_results)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
