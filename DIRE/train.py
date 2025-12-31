from utils.config import cfg as cfg_base  # isort: split

import os
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger

from copy import deepcopy


def main(train_set, test_set, eval_type):
    start = time.time()
    print("Start Time: " + str(time.localtime()))

    cfg = deepcopy(cfg_base)
    cfg.datasets = [f"GenImage-{train_set}"]
    if eval_type == "hard":
        cfg.datasets_test = [f"GenImage-{test_set}-hard"]
    else:
        cfg.datasets_test = [f"GenImage-{test_set}"]

    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
    data_loader = create_dataloader(cfg)
    dataset_size = len(data_loader)

    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    # Train
    trainer = Trainer(cfg)
    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(cfg.nepoch):
        epoch_iter = 0
        for data in tqdm(data_loader, dynamic_ncols=True):
            trainer.total_steps += 1
            epoch_iter += cfg.batch_size

            trainer.set_input(data)
            trainer.optimize_parameters()
            train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

            if trainer.total_steps % cfg.save_latest_freq == 0:
                log.write(
                    "saving the latest model %s (epoch %d, model.total_steps %d)\n"
                    % (cfg.exp_name, epoch, trainer.total_steps)
                )
                trainer.save_networks("latest")

        if epoch % cfg.save_epoch_freq == 0:
            log.write("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
            trainer.save_networks("latest")
            trainer.save_networks(epoch)

        starteval = time.time()
        print("Start Eval Time: " + str(time.localtime()))

        # Validation
        trainer.eval()
        val_results = validate(trainer.model, val_cfg)
        val_writer.add_scalar("AP", val_results["AP"], trainer.total_steps)
        val_writer.add_scalar("ACC", val_results["ACC"], trainer.total_steps)
        log.write(f"(Val @ epoch {epoch}) AP: {val_results['AP']}; ACC: {val_results['ACC']}\n")

        print("End Eval Time: " + str(time.localtime()))
        print("Elapsed Eval Secs: " + str(time.time() - starteval))

        if cfg.earlystop:
            early_stopping(val_results["ACC"], trainer)
            if early_stopping.early_stop:
                if trainer.adjust_learning_rate():
                    log.write("Learning rate dropped by 10, continue training...\n")
                    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    log.write("Early stopping.\n")
                    break
        if cfg.warmup:
            trainer.scheduler.step()
        trainer.train()

    print("End Time: " + str(time.localtime()))
    print("Elapsed Secs: " + str(time.time() - start))

    # Test
    # trainer = Trainer(cfg)
    # load_model_epoch = 10
    # trainer.load_networks(load_model_epoch)
    # trainer.eval()
    # val_results = validate(trainer.model, val_cfg)
    # val_writer.add_scalar("AP", val_results["AP"], trainer.total_steps)
    # val_writer.add_scalar("ACC", val_results["ACC"], trainer.total_steps)
    # log.write(f"(Val @ epoch {load_model_epoch}) AP: {val_results['AP']}; ACC: {val_results['ACC']}\n")

if __name__ == "__main__":
    # main("GLIDE", "GLIDE", "base")

    for ds in ["Midjourney", "SDV14", "SDV15", "ADM", "GLIDE", "Wukong", "VQDM", "BigGAN"]:
        print(f"\n\n ------------- Cross-Validation SDV1.4 -> {ds} --------------\n\n")
        main("SDV14", ds, "base")
        main("SDV14", ds, "hard")

