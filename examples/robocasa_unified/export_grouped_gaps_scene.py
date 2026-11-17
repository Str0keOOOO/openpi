"""Verify CPU gap candidates in the official environment, exporting to staging only."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from grouped import apply, dump, images_and_state, restore, snapshot, validate_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--search', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--scene', type=int, required=True)
    p.add_argument('--allow-resampling', action='store_true')
    a = p.parse_args()
    relative = Path('scenes') / a.task / f'scene_{a.scene:02d}'
    source = a.dataset / relative
    destination = a.search / 'official_verified' / relative
    destination.mkdir(parents=True, exist_ok=True)
    env = None
    results = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'robocasa_augmentation'))
        from gpu_routing import egl_device_for_uuid
        egl = egl_device_for_uuid(os.environ['CUDA_VISIBLE_DEVICES'])
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(egl)
        from sample_grouped_scene import make_env
        config = json.loads((a.dataset / 'config.json').read_text())
        anchor = json.loads((source / 'anchor/anchor.json').read_text())
        with np.load(source / 'anchor/snapshot.npz') as z:
            baseline = {k: z[k].copy() for k in z.files}
        env = make_env(a.task, anchor['scene'], anchor['scene']['seed'], egl)
        restore(env, anchor, baseline, reload_model=True)
        for marker in sorted((a.search / relative).glob('*/search_result.json')):
            search = json.loads(marker.read_text())
            if search['status'] != 'legal_candidate_pending_official_export':
                continue
            case = search['case']
            dest = destination / case['id']
            try:
                original = json.loads((source / 'starts' / case['id'] / 'configuration.json').read_text())
                source_case = search.get('original_case', case)
                if original['accepted'] or original['case'] != source_case:
                    raise RuntimeError('Source slot changed during gap search')
                if case != source_case:
                    if not a.allow_resampling:
                        raise RuntimeError('Resampled signs/bins require explicit resampling mode')
                    for key in set(case) | set(source_case):
                        if key not in ('signs', 'magnitude_subbins') and case.get(key) != source_case.get(key):
                            raise RuntimeError('Resampling changed scene, family, tier or identity')
                restore(env, anchor, baseline)
                requested = np.asarray(search['actual_delta'])
                actual, legality = apply(env, requested)
                if legality['status'] != 'ready' or not validate_delta(config, case, actual):
                    raise RuntimeError('Official collision/limit/stratum check failed: ' + str(legality))
                with np.load(marker.parent / 'candidate_state.npz') as z:
                    cpu_error = float(np.max(np.abs(env.sim.data.qpos - z['qpos'])))
                if cpu_error > 1e-7:
                    raise RuntimeError('CPU/official qpos mismatch: ' + str(cpu_error))
                saved = snapshot(env)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / 'state.npz', **saved)
                state, paths, cameras = images_and_state(env, dest / 'images')
                # Per-case state/RGB roundtrip; fresh XML restoration is also checked
                # independently for every changed scene before publication.
                restore(env, anchor, saved, reload_model=not a.allow_resampling)
                state2, paths2, _ = images_and_state(env, dest / 'roundtrip')
                import imageio.v2 as imageio
                errors = {cam: float(np.abs(imageio.imread(paths[cam]).astype(float) -
                                            imageio.imread(paths2[cam]).astype(float)).mean())
                          for cam in paths}
                if not np.allclose(state, state2, atol=1e-7, rtol=0) or max(errors.values()) > .5:
                    raise RuntimeError('Independent RGB/State restoration failed: ' + str(errors))
                _, recheck = apply(env, np.zeros(11))
                if recheck['status'] != 'ready':
                    raise RuntimeError('Exported snapshot failed official collision check')
                record = dict(original, case=case, accepted=True, actual_delta=actual.tolist(),
                              requested_delta=requested.tolist(), state_file=str(dest / 'state.npz'),
                              observation_state=state.tolist(), images=paths, cameras=cameras,
                              legality=legality, gapfill=True, reason=None,
                              gapfill_search_result=str(marker), policy_actions_executed=0)
                if a.allow_resampling:
                    record.update(sampling_mode='same_level_legal_resampling', original_sampling_case=source_case)
                dump(dest / 'configuration.json', record)
                result = dict(id=case['id'], passed=True, cpu_qpos_error=cpu_error,
                              rgb_mae_255=errors, actual_delta=actual.tolist(),
                              configuration_file=str(dest / 'configuration.json'))
            except Exception:
                result = dict(id=case['id'], passed=False, error=traceback.format_exc())
            dump(dest / 'validation.json', result)
            results.append(result)
            print(a.task, a.scene, case['id'], result['passed'], flush=True)
        dump(destination / 'summary.json', dict(task=a.task, scene_index=a.scene,
             passed=all(x['passed'] for x in results), results=results, policy_actions_executed=0))
    except Exception:
        dump(destination / 'summary.json', dict(passed=False, error=traceback.format_exc(), results=results))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
