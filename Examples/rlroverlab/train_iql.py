import argparse

from common import (
    FEEDFORWARD_ACTOR_CONFIG,
    FEEDFORWARD_VALUE_CONFIG,
    add_dataset_args,
    add_eval_args,
    add_frame_stack_args,
    add_trainer_args,
    build_module,
    checkpoint_dir_from_wandb,
    configure_wandb_env,
    count_parameters,
    load_json_config,
    run_eval_after_train,
    save_export_config,
)


parser = argparse.ArgumentParser("Train CloneLab IQL on an RLRoverLab HDF5 dataset.")
add_dataset_args(parser)
add_trainer_args(parser, batch_size=100, epochs=20)
add_frame_stack_args(parser)
add_eval_args(parser)
parser.add_argument("--actions_lr", type=float, default=3e-4, help="Actor learning rate.")
parser.add_argument("--value_lr", type=float, default=3e-4, help="Value learning rate.")
parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic learning rate.")
parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--tau", type=float, default=0.01, help="Target network update rate.")
parser.add_argument("--expectile", type=float, default=0.8, help="IQL expectile.")
parser.add_argument("--temperature", type=float, default=0.0, help="Advantage temperature.")
parser.add_argument("--target_update_freq", type=int, default=1, help="Target update frequency.")
parser.add_argument("--actor_factory", type=str, default="Examples.isaaclab.models_cai:actor_gaussian_image")
parser.add_argument("--critic_factory", type=str, default="Examples.isaaclab.models_cai:TwinQ_image")
parser.add_argument("--value_factory", type=str, default="Examples.isaaclab.models_cai:v_image")
parser.add_argument("--actor_config", type=str, default=None, help="Optional JSON actor config.")
parser.add_argument("--critic_config", type=str, default=None, help="Optional JSON critic config.")
parser.add_argument("--value_config", type=str, default=None, help="Optional JSON value config.")


def main() -> None:
    configure_wandb_env(args)

    from CloneRL.algorithms.torch.offline_rl.iql import IQL
    from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDatasetRandom
    from CloneRL.trainers.torch.sequential import SequentialTrainer

    actor_config = load_json_config(args.actor_config, FEEDFORWARD_ACTOR_CONFIG)
    critic_config = load_json_config(args.critic_config, FEEDFORWARD_ACTOR_CONFIG)
    value_config = load_json_config(args.value_config, FEEDFORWARD_VALUE_CONFIG)
    if args.frame_stacking and args.actor_config is None:
        actor_config["image_channels"] *= args.num_stacked_frames
        actor_config["depth_channels"] *= args.num_stacked_frames
    if args.frame_stacking and args.critic_config is None:
        critic_config["image_channels"] *= args.num_stacked_frames
        critic_config["depth_channels"] *= args.num_stacked_frames
    if args.frame_stacking and args.value_config is None:
        value_config["image_channels"] *= args.num_stacked_frames
        value_config["depth_channels"] *= args.num_stacked_frames

    actor = build_module(args.actor_factory, actor_config, args.device)
    critic = build_module(args.critic_factory, critic_config, args.device)
    value = build_module(args.value_factory, value_config, args.device)
    print(f"Actor parameters: {count_parameters(actor):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")
    print(f"Value parameters: {count_parameters(value):,}")

    iql_config = {
        "actions_lr": args.actions_lr,
        "value_lr": args.value_lr,
        "critic_lr": args.critic_lr,
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "temperature": args.temperature,
        "target_update_freq": args.target_update_freq,
    }
    agent = IQL(actor_policy=actor, value_policy=value, critic_policy=critic, cfg=iql_config, **iql_config)
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
    args = parser.parse_args()
    main()
