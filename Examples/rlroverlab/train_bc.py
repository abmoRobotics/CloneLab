import argparse

from common import (
    FEEDFORWARD_ACTOR_CONFIG,
    add_dataset_args,
    add_eval_args,
    add_frame_stack_args,
    add_trainer_args,
    build_module,
    checkpoint_dir_from_wandb,
    configure_wandb_env,
    count_parameters,
    run_eval_after_train,
    save_export_config,
)
from run_config import config_path, load_model_config, parse_args_with_config


parser = argparse.ArgumentParser("Train CloneLab BC on an RLRoverLab HDF5 dataset.")
add_dataset_args(parser)
add_trainer_args(parser, batch_size=64, epochs=30)
add_frame_stack_args(parser)
add_eval_args(parser)
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Actor learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Actor weight decay.")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
parser.add_argument(
    "--actor_factory",
    type=str,
    default="CloneRL.models.torch:actor_gaussian_image",
    help="Import path for the actor factory.",
)
parser.add_argument("--actor_config", type=str, default=None, help="Optional JSON/YAML actor config.")


def main() -> None:
    configure_wandb_env(args)

    from CloneRL.algorithms.torch.imitation_learning.bc import BehaviourCloning
    from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDatasetRandom
    from CloneRL.trainers.torch.sequential import SequentialTrainer

    actor_config = load_model_config(args.actor_config, FEEDFORWARD_ACTOR_CONFIG)
    if args.frame_stacking:
        actor_config["image_channels"] *= args.num_stacked_frames
        actor_config["depth_channels"] *= args.num_stacked_frames

    actor = build_module(args.actor_factory, actor_config, args.device)
    print(f"Actor parameters: {count_parameters(actor):,}")

    agent = BehaviourCloning(
        actor_policy=actor,
        cfg={"lr": args.learning_rate, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip},
        device=args.device,
    )
    checkpoint_dir = checkpoint_dir_from_wandb()
    save_export_config(checkpoint_dir, "feedforward", args.actor_factory, actor_config)

    dataset = HDF5DictDatasetRandom(
        args.dataset,
        min_idx=args.min_idx,
        max_idx=args.max_idx,
        total_samples=args.total_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        use_frame_stacking=args.frame_stacking,
        frame_stack_stride=args.frame_stack_stride,
        num_stacked_frames=args.num_stacked_frames,
    )
    val_dataset = HDF5DictDatasetRandom(
        args.val_dataset or args.dataset,
        min_idx=args.val_min_idx,
        max_idx=args.val_max_idx,
        total_samples=args.val_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        use_frame_stacking=args.frame_stacking,
        frame_stack_stride=args.frame_stack_stride,
        num_stacked_frames=args.num_stacked_frames,
    )

    trainer = SequentialTrainer(
        cfg={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "save_freq": args.save_freq,
            "validation_freq": args.validation_freq,
            "log_freq": args.log_freq,
        },
        policy=agent,
        dataset=dataset,
        val_dataset=val_dataset,
    )
    trainer.train()
    run_eval_after_train(args, checkpoint_dir)


if __name__ == "__main__":
    args = parse_args_with_config(parser, default_config=config_path("bc.yaml"), required=("dataset",))
    main()
