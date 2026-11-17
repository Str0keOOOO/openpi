"""Independent JAX policy service with an explicit seed for each action chunk."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-config", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = policy_config.create_trained_policy(config.get_config(a.policy_config), str(a.checkpoint))
    if policy._is_pytorch_model:
        raise ValueError("This experiment uses the project JAX pi05 implementation")

    class Seeded:
        def infer(self, observation):
            obs = dict(observation)
            if "__inference_seed" not in obs:
                raise ValueError("Missing explicit per-chunk inference seed")
            seed = int(obs.pop("__inference_seed"))
            policy._rng = jax.random.key(seed)
            return policy.infer(obs)

    WebsocketPolicyServer(
        Seeded(),
        host="127.0.0.1",
        port=a.port,
        metadata=dict(
            explicit_inference_seed=True, checkpoint=str(a.checkpoint.resolve()), policy_config=a.policy_config
        ),
    ).serve_forever()


if __name__ == "__main__":
    main()
