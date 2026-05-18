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
    load_json_config,
    run_eval_after_train,
)


parser = argparse.ArgumentParser("Train CloneLab recurrent BC on an RLRoverLab HDF5 dataset.")
add_dataset_args(parser)
add_trainer_args(parser, batch_size=16, epochs=12)
add_eval_args(parser, recurrent=True)
parser.add_argument("--sequence_length", type=int, default=16, help="Training sequence length.")
parser.add_argument(
    "--image_mode",
    type=str,
    default="rgb",
    help="rgb, grayscale, depth, rgbd, grayscale_depth, or none.",
)
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Actor learning rate.")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Actor weight decay.")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
parser.add_argument("--lr_scheduler", type=str, default=None, help="Optional LR scheduler: step, cosine, or none.")
parser.add_argument("--lr_decay_steps", type=int, default=1000, help="Step scheduler decay interval.")
parser.add_argument("--lr_decay_rate", type=float, default=0.99, help="Step scheduler decay rate.")
parser.add_argument("--actor_factory", type=str, default="Examples.isaaclab.models_cai:GRUActorGaussian")
parser.add_argument("--actor_config", type=str, default=None, help="Optional JSON actor config.")


def main() -> None:
    configure_wandb_env(args)

    from CloneRL.algorithms.torch.imitation_learning.bc import BehaviourCloningRNN
    from CloneRL.dataloader.hdf import HDF5RandomSequenceGRUDataset
    from CloneRL.trainers.torch.sequential_recurrent import SequentialRecurrentTrainer

    base_config = dict(RECURRENT_BASE_CONFIG)
    base_config["device"] = args.device
    base_config["image_channels"] = image_channels_for_mode(args.image_mode)
    actor_config = load_json_config(args.actor_config, {**base_config, "action_dim": 2})

    actor = build_module(args.actor_factory, actor_config, args.device)
    print(f"Actor parameters: {count_parameters(actor):,}")

    bc_config = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "lr_scheduler": args.lr_scheduler,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_decay_rate": args.lr_decay_rate,
        "reset_hidden_on_done": True,
    }
    agent = BehaviourCloningRNN(actor_policy=actor, cfg=bc_config, device=args.device, **bc_config)
    checkpoint_dir = checkpoint_dir_from_wandb()

    dataset = HDF5RandomSequenceGRUDataset(
        args.dataset,
        sequence_length=args.sequence_length,
        min_idx=args.min_idx,
        max_idx=args.max_idx,
        total_samples=args.total_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        image_mode=args.image_mode,
    )
    val_dataset = HDF5RandomSequenceGRUDataset(
        args.val_dataset or args.dataset,
        sequence_length=args.sequence_length,
        min_idx=args.val_min_idx,
        max_idx=args.val_max_idx,
        total_samples=args.val_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        image_mode=args.image_mode,
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
    args = parser.parse_args()
    main()
