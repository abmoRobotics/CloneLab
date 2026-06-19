import argparse

from common import (
    RECURRENT_BASE_CONFIG,
    add_dataset_args,
    add_eval_args,
    add_trainer_args,
    build_module,
    checkpoint_dir_from_wandb,
    configure_wandb_env,
    count_parameters,
    image_channels_for_mode,
    run_eval_after_train,
    save_export_config,
)
from run_config import config_path, load_model_config, parse_args_with_config


parser = argparse.ArgumentParser("Train risk-conditioned recurrent IQL on cached DINO/DA3 sequences.")
add_dataset_args(parser)
add_trainer_args(parser, batch_size=32, epochs=5)
add_eval_args(parser, recurrent=True)
parser.add_argument("--sequence_length", type=int, default=8, help="Training sequence length.")
parser.add_argument("--image_mode", type=str, default="none", help="Kept for recurrent trainer compatibility.")
parser.add_argument(
    "--clearance_path",
    type=str,
    default="transitions/extra/risk/min_distance_to_rock",
    help="Transition-level obstacle clearance dataset.",
)
parser.add_argument("--actions_lr", type=float, default=1e-4, help="Actor learning rate.")
parser.add_argument("--value_lr", type=float, default=1e-4, help="Value learning rate.")
parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic learning rate.")
parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate.")
parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile.")
parser.add_argument("--temperature", type=float, default=1.0, help="Advantage temperature.")
parser.add_argument("--target_update_freq", type=int, default=1, help="Target update frequency.")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
parser.add_argument("--lambda_risk", type=float, default=0.02, help="Maximum clearance-cost coefficient.")
parser.add_argument("--d_ref", type=float, default=5.0, help="Reference clearance in meters.")
parser.add_argument("--clearance_exponent", type=float, default=2.5, help="Clearance cost exponent.")
parser.add_argument("--alpha_min", type=float, default=0.0, help="Minimum sampled risk preference.")
parser.add_argument("--alpha_max", type=float, default=1.0, help="Maximum sampled risk preference.")
parser.add_argument(
    "--actor_factory",
    type=str,
    default="CloneRL.models.torch:risk_dino_da_iql_actor",
)
parser.add_argument(
    "--critic_factory",
    type=str,
    default="CloneRL.models.torch:risk_dino_da_iql_twin_q",
)
parser.add_argument(
    "--value_factory",
    type=str,
    default="CloneRL.models.torch:risk_dino_da_iql_value",
)
parser.add_argument("--actor_config", type=str, default=None, help="Optional JSON/YAML actor config.")
parser.add_argument("--critic_config", type=str, default=None, help="Optional JSON/YAML critic config.")
parser.add_argument("--value_config", type=str, default=None, help="Optional JSON/YAML value config.")


def _build_dataset(
    file_path: str,
    *,
    min_idx: int,
    max_idx: int | None,
    total_samples: int,
    actor_config: dict,
):
    from CloneRL.dataloader.hdf import RLRoverLabRiskDinoDARandomSequenceDataset

    print(f"[INFO] Loading risk-aware cached DINO/DA sequence dataset: {file_path}")
    return RLRoverLabRiskDinoDARandomSequenceDataset(
        file_path,
        clearance_path=args.clearance_path,
        sequence_length=args.sequence_length,
        min_idx=min_idx,
        max_idx=max_idx,
        total_samples=total_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        return_next_obs=True,
        expected_da_shape=(
            int(actor_config.get("depth_channels", 1)),
            int(actor_config.get("depth_height", 72)),
            int(actor_config.get("depth_width", 128)),
        ),
    )


def main() -> None:
    configure_wandb_env(args)

    from CloneRL.algorithms.torch.offline_rl.iql import RiskConditionedIQLRecurrent
    from CloneRL.trainers.torch.sequential_recurrent import SequentialRecurrentTrainer

    base_config = dict(RECURRENT_BASE_CONFIG)
    base_config["device"] = args.device
    base_config["image_channels"] = image_channels_for_mode(args.image_mode)
    actor_config = load_model_config(args.actor_config, {**base_config, "action_dim": 2})
    critic_config = load_model_config(args.critic_config, {**base_config, "action_dim": 2})
    value_config = load_model_config(args.value_config, base_config)

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
        "grad_clip": args.grad_clip,
        "reset_hidden_on_done": True,
        "lambda_risk": args.lambda_risk,
        "d_ref": args.d_ref,
        "clearance_exponent": args.clearance_exponent,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
    }
    agent = RiskConditionedIQLRecurrent(
        actor_policy=actor,
        value_policy=value,
        critic_policy=critic,
        cfg=iql_config,
        device=args.device,
        **iql_config,
    )
    checkpoint_dir = checkpoint_dir_from_wandb()
    save_export_config(checkpoint_dir, "recurrent", args.actor_factory, actor_config)

    dataset = _build_dataset(
        args.dataset,
        min_idx=args.min_idx,
        max_idx=args.max_idx,
        total_samples=args.total_samples,
        actor_config=actor_config,
    )
    val_dataset = _build_dataset(
        args.val_dataset or args.dataset,
        min_idx=args.val_min_idx,
        max_idx=args.val_max_idx,
        total_samples=args.val_samples,
        actor_config=actor_config,
    )

    trainer = SequentialRecurrentTrainer(
        cfg={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "save_freq": args.save_freq,
            "validation_freq": args.validation_freq,
            "log_freq": args.log_freq,
            "sequence_length": args.sequence_length,
            "image_mode": args.image_mode,
        },
        policy=agent,
        dataset=dataset,
        val_dataset=val_dataset,
    )
    trainer.train()
    run_eval_after_train(args, checkpoint_dir)


if __name__ == "__main__":
    args = parse_args_with_config(
        parser,
        default_config=config_path("rc_iql_dino_da_recurrent_h256_5ep.yaml"),
        required=("dataset",),
    )
    main()

