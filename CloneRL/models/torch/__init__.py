from CloneRL.models.torch.feedforward import (
    GaussianImageActor,
    ImageQ,
    ImageValue,
    TwinImageQ,
)
from CloneRL.models.torch.recurrent import GRUActorGaussian, GRUQNetwork, GRUTwinQ, GRUValue

actor_gaussian_image = GaussianImageActor
v_image = ImageValue
q_image = ImageQ
twin_q_image = TwinImageQ
TwinQ_image = TwinImageQ
gru_actor_gaussian = GRUActorGaussian
gru_value = GRUValue
gru_twin_q = GRUTwinQ

__all__ = [
    "GaussianImageActor",
    "ImageQ",
    "ImageValue",
    "TwinImageQ",
    "GRUActorGaussian",
    "GRUQNetwork",
    "GRUTwinQ",
    "GRUValue",
    "actor_gaussian_image",
    "q_image",
    "twin_q_image",
    "TwinQ_image",
    "gru_actor_gaussian",
    "gru_twin_q",
    "gru_value",
    "v_image",
]
