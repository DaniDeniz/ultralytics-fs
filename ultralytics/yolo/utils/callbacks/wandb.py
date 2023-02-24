from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    wandb = None

log_dict = {}


def on_pretrain_routine_start(trainer):
    wandb.init(project=trainer.args.project or "YOLOv8",
               name=trainer.args.name,
               resume="allow",
               tags=['YOLOv8'],
               config=vars(trainer.args))


def on_train_epoch_end(trainer):
    if trainer.epoch == 1:
        log_dict['Mosaics'] = [wandb.Image(str(f)) for f in trainer.save_dir.glob('train_batch*.jpg')]


def on_fit_epoch_end(trainer):
    if trainer.epoch == 0:
        model_info = {
            "Parameters": get_num_params(trainer.model),
            "GFLOPs": round(get_flops(trainer.model), 3),
            "Inference speed (ms/img)": round(trainer.validator.speed[1], 3)}
        wandb.run.summary.update(model_info)
    keys = [
        'train/box_loss',
        'train/cls_loss',
        'train/dfl_loss',  # train loss
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',  # metrics
        'val/box_loss',
        'val/cls_loss',
        'val/dfl_loss',  # val loss
        'x/lr0',
        'x/lr1',
        'x/lr2']
    # if trainer.epoch != trainer.epochs - 1:
    log_vals = list(trainer.tloss) + list(trainer.metrics.values()) + [x["lr"] for x in trainer.optimizer.param_groups]
    x = dict(zip(keys, log_vals))
    for key, value in x.items():
        log_dict[key] = value
    if trainer.best_fitness == trainer.fitness:
        best_results = [trainer.epoch] + log_vals[3:7]
        for i, name in enumerate(['best/epoch', 'best/precision', 'best/recall', 'best/mAP50', 'best/mAP50-95']):
            wandb.run.summary[name] = best_results[i]
    wandb.log(log_dict)
    log_dict.clear()


def on_train_end(trainer):
    files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
    files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]
    log_dict.clear()
    log_dict['Results'] = [wandb.Image(str(f)) for f in files]
    wandb.log(log_dict)
    wandb.log_artifact(str(trainer.best if trainer.best.exists() else trainer.last),
                       type='model',
                       name=f'run_{wandb.run.id}_model',
                       aliases=['latest', 'best', 'stripped'])
    log_dict.clear()


def teardown(trainer):
    wandb.run.finish()


def on_val_end(validator):
    files = sorted(validator.save_dir.glob('val*.jpg'))
    if files and validator.training:
        log_dict.clear()
        log_dict['Validation'] = [wandb.Image(str(f), caption=f.name) for f in files]


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_train_end": on_train_end,
    "on_val_end": on_val_end,
    "teardown": teardown,} if wandb else {}