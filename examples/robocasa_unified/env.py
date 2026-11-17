"""Create the RoboCasa environment used by the fixed test set."""

from grouped import CAMERAS


def make_env(task, pair, seed, gpu=-1, render=True):
    import robocasa, robosuite
    from robosuite.controllers import load_composite_controller_config

    controller = load_composite_controller_config(controller=None, robot="PandaOmron")
    return robosuite.make(
        env_name=task,
        robots="PandaOmron",
        controller_configs=controller,
        layout_ids=[pair["layout_id"]],
        style_ids=[pair["style_id"]],
        obj_instance_split="pretrain",
        seed=seed,
        randomize_cameras=False,
        initialization_noise=None,
        robot_spawn_deviation_pos_x=0.0,
        robot_spawn_deviation_pos_y=0.0,
        robot_spawn_deviation_rot=0.0,
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=False,
        use_object_obs=True,
        camera_names=list(CAMERAS),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=gpu,
        ignore_done=True,
    )
