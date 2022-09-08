import os
import argparse
import json
import wandb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.dptnetq import DPTNetQ
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


def combiner_init(x, n_bits=8, sign=True):
    min_range, max_range = x.min(), x.max()
    t = torch.maximum(torch.abs(min_range), torch.abs(max_range))
    delta = t / (2 ** (n_bits - int(sign)))
    X = torch.round(x / delta)
    Qmin = -2 ** (n_bits - 1) if sign else 0
    Qmax = 2 ** (n_bits - 1) - 1 if sign else 2 ** n_bits - 1
    x_quant = delta * torch.clip(X, Qmin, Qmax)
    w = t*(x-x_quant)/(0.5*delta) # Quantization error with ML align
    return w


def get_pretrain_pytorch_model(model, weights_path, splitter=False, combiner=False):
    model_state_dict = model.state_dict()
    model_state_dict_weights = torch.load(weights_path).get('state_dict')
    for new_key,key in zip(model_state_dict.keys(),model_state_dict_weights.keys()):
        if splitter and new_key == "encoder.0.weight":
            x = model_state_dict_weights.get(key)
            y = x.repeat(1, 2, 1)
            y[:, 1:, :] = torch.mean(x) + torch.std(x)*torch.randn_like(x) # gaussian
            model_state_dict[new_key] = y
            print("Splitter initialiation is done!")
        elif combiner and new_key == "decoder.weight":
            x = model_state_dict_weights.get(key)
            y = x.repeat(1, 2, 1)
            y[:, 1:, :] = combiner_init(x)
            model_state_dict[new_key] = y
            print("Combiner initialiation is done!")
        else:
            model_state_dict[new_key] = model_state_dict_weights.get(key)
    model.load_state_dict(model_state_dict, strict=True)
    return model


# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(conf):

    seed = conf["training"].get("seed", 0)
    torch.manual_seed(seed)

    train_set = LibriMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    val_set = LibriMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    # ------ QAT model --------------
    pretrained = conf["training"].get("pretrained", None)
    enc_num_ch = conf["filterbank"].get("enc_num_ch", 1)
    dec_num_ch = conf["filterbank"].get("dec_num_ch", 1)
    qat = conf["training"]["qat"]
    kd_lambda = conf["training"].get("kd_lambda", 0)

    model = DPTNetQ(n_src=conf["data"]["n_src"],
                    enc_num_ch=enc_num_ch,
                    dec_num_ch=dec_num_ch)

    float_model = DPTNetQ(n_src=conf["data"]["n_src"])

    if pretrained is not None:
        model = get_pretrain_pytorch_model(model, pretrained, splitter=enc_num_ch>1, combiner=dec_num_ch>1)
        if kd_lambda > 0:
            float_model = get_pretrain_pytorch_model(float_model, pretrained)
            model.to(DEVICE)
            model.eval()

    if qat:
        weight_quant = conf["training"].get("weight_quant", True)
        act_quant = conf["training"].get("act_quant", True)
        in_quant = conf["training"].get("in_quant", False)
        out_quant = conf["training"].get("out_quant", True)
        model.quantize_model(qat_weight_quant=weight_quant, qat_act_quant=act_quant, in_quant=in_quant, out_quant=out_quant)
        model.to(DEVICE)
    # -------------------------------

    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # WandB
    if conf["training"]["wandb"]:
        print("WandB is enable!")
        test_name = exp_dir.split('/')[-1]
        PROJECT_NAME = "DPTNet_"+conf["data"]["task"]
        wandb.init(project=PROJECT_NAME, name=test_name, dir=exp_dir)
        wandb.finish()
        wandbLogger = WandbLogger(project=PROJECT_NAME, name=test_name, log_model='all')

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        float_model=float_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
        kd_lambda=kd_lambda,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True)
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    if conf["training"]["wandb"]:
        trainer = pl.Trainer(
            max_epochs=conf["training"]["epochs"],
            callbacks=callbacks,
            default_root_dir=exp_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp",
            devices="auto",
            limit_train_batches=1.0,  # Useful for fast experiment
            gradient_clip_val=5.0,
            logger=wandbLogger,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=conf["training"]["epochs"],
            callbacks=callbacks,
            default_root_dir=exp_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp",
            devices="auto",
            limit_train_batches=1.0,  # Useful for fast experiment
            gradient_clip_val=5.0,
        )

    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save latest model
    torch.save(system.model.state_dict(), os.path.join(exp_dir, "latest_model.pth"))

    # Save best model
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    torch.save(system.model.state_dict(), os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf_sony.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
