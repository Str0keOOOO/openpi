"""Official verification and transactional publication of quota-preserving gap fills."""
import argparse
import collections
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from grouped import dump

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--search', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--render-gpus', type=int, nargs='+')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--allow-resampling', action='store_true')
    a = p.parse_args()
    lock = (a.search / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)  # Never publish while CPU workers read the frozen source.
    published = a.search / 'publication.json'
    if published.exists() and json.loads(published.read_text()).get('status') == 'published':
        print(published.read_text(), flush=True)
        return
    old_hash = hashlib.sha256((a.dataset / 'frozen_manifest.json').read_bytes()).hexdigest()
    if (a.dataset / 'FROZEN').read_text().strip() != old_hash:
        raise RuntimeError('Source freeze marker is inconsistent')
    if json.loads((a.search / 'config.json').read_text())['source_manifest_sha256'] != old_hash:
        raise RuntimeError('The frozen source changed after search')
    searches = [json.loads(x.read_text()) for x in a.search.glob('scenes/*/scene_*/*/search_result.json')]
    expected = json.loads((a.dataset / 'sampling_summary.json').read_text())['invalid']
    errors = [str(x) for x in a.search.glob('scenes/*/scene_*/summary.json')
              if json.loads(x.read_text()).get('status') == 'error']
    if len(searches) != expected or errors:
        raise RuntimeError(f'Incomplete gap search: {len(searches)}/{expected}, errors={errors}')
    candidates = [x for x in searches if x['status'] == 'legal_candidate_pending_official_export']
    search_config = json.loads((a.search / 'config.json').read_text())
    if a.allow_resampling and search_config.get('sampling_mode') != 'same_level_legal_resampling':
        raise RuntimeError('Resampling mode requires an explicitly resampled candidate collection')
    gpu_ids = a.render_gpus or [a.gpu]
    envs = []
    for gpu in gpu_ids:
        response = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
                                            '--format=csv,noheader,nounits'], text=True).strip()
        uuid, free = response.split(',')
        if int(free) < 4000:
            raise RuntimeError('Insufficient free memory for independent rendering on GPU '+str(gpu))
        envs.append(dict(os.environ, CUDA_VISIBLE_DEVICES=uuid.strip(), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                   __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))
    def export(job):
        index, (task, scene) = job
        folder = a.search / 'official_verified/scenes' / task / f'scene_{scene:02d}'
        wanted = {x['id'] for x in candidates if (x['task'], x['scene_index']) == (task, scene)}
        summary = folder / 'summary.json'
        existing = json.loads(summary.read_text()) if summary.exists() else {}
        if existing.get('passed') and {x['id'] for x in existing['results']} == wanted:
            return
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'run.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(HERE / 'export_grouped_gaps_scene.py'),
                            '--dataset', str(a.dataset), '--search', str(a.search),
                            '--task', task, '--scene', str(scene)]+(['--allow-resampling'] if a.allow_resampling else []), env=envs[index%len(envs)], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
        print('Official export completed:', task, scene, flush=True)
    jobs = sorted({(x['task'], x['scene_index']) for x in candidates})
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(export, enumerate(jobs)))
    verified = []
    for x in candidates:
        relative = Path('scenes') / x['task'] / f"scene_{x['scene_index']:02d}"
        folder = a.search / 'official_verified' / relative / x['id']
        result = json.loads((folder / 'validation.json').read_text())
        if result['passed']:
            verified.append((relative, folder))
    if not verified:
        dump(published, dict(status='no_officially_valid_additions', tested=len(searches),
                             remaining=expected, original_unchanged=True))
        return
    # Full independent copy; hardlinks would corrupt the backup when CSVs are rewritten.
    backup = a.search / 'before_publication_backup'
    if not backup.exists():
        partial = a.search / 'before_publication_backup.partial'
        shutil.copytree(a.dataset, partial, dirs_exist_ok=True)
        partial.rename(backup)
    dump(published, dict(status='publishing', previous_manifest_sha256=old_hash,
                         additions=len(verified), backup=str(backup)))
    (a.dataset / 'FROZEN').unlink()
    try:
        changed = set()
        for relative, folder in verified:
            r = json.loads((folder / 'configuration.json').read_text())
            dest = a.dataset / relative / 'starts' / r['case']['id']
            shutil.copy2(folder / 'state.npz', dest / 'state.gapfill.npz')
            shutil.copytree(folder / 'images', dest / 'gapfill_images', dirs_exist_ok=True)
            r.update(state_file=str(dest / 'state.gapfill.npz'),
                     images={cam: str(dest / 'gapfill_images' / Path(path).name)
                             for cam, path in r['images'].items()})
            dump(dest / 'configuration.json', r)
            changed.add(relative)
        for relative in sorted(changed):
            scene = a.dataset / relative
            rows = [json.loads(x.read_text()) for x in (scene / 'starts').glob('*/configuration.json')]
            summary = json.loads((scene / 'summary.json').read_text())
            summary.update(valid=sum(x['accepted'] for x in rows), invalid=sum(not x['accepted'] for x in rows))
            dump(scene / 'summary.json', summary)
        def validate(job):
            index, relative = job
            scene = a.dataset / relative
            with (scene / 'gapfill_validation.log').open('w') as log:
                subprocess.run([sys.executable, '-u', str(HERE / 'validate_grouped_scene.py'),
                                '--output', str(a.dataset), '--task', relative.parts[1],
                                '--scene', str(int(relative.parts[2].split('_')[1])), '--gpu', str(a.gpu)],
                               env=envs[index%len(envs)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            if not json.loads((scene / 'restoration_validation.json').read_text())['passed']:
                raise RuntimeError('Fresh official validation failed: ' + str(scene))
            print('Fresh scene validation completed:', relative, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(validate, enumerate(sorted(changed))))
        from build_grouped import summarize
        c = json.loads((a.dataset / 'config.json').read_text())
        if a.allow_resampling:
            c.update(replacement_sampling='whole_tier_legal_points', marginal_balance='soft_target_over_physically_valid_points')
            dump(a.dataset / 'config.json', c)
            from grouped import digest
            (a.dataset / 'test_signature.txt').write_text(digest({k:v for k,v in c.items() if k!='models'}))
        result = summarize(c, a.dataset, final=True)
        checks = [json.loads(x.read_text()) for x in a.dataset.glob('scenes/*/scene_*/restoration_validation.json')]
        if len(checks) != 20 or not all(x['passed'] for x in checks):
            raise RuntimeError('Not all scenes retain valid restoration checks')
        new_hash = hashlib.sha256((a.dataset / 'frozen_manifest.json').read_bytes()).hexdigest()
        validation = dict(passed=True, scenes_checked=20,
                          valid_starts_checked=sum(x['valid_starts_checked'] for x in checks),
                          manifest_sha256=new_hash, policy_actions_executed=0, errors=[],
                          previous_manifest_sha256=old_hash, newly_revalidated_scenes=[str(x) for x in sorted(changed)],
                          unchanged_scenes_validation_reused=True)
        if validation['valid_starts_checked'] != result['valid']:
            raise RuntimeError('Validation and manifest counts differ')
        dump(a.dataset / 'validation_summary.json', validation)
        balance = json.loads((a.dataset / 'requested_balance_validation.json').read_text())
        balance['legal_subset_count'] = result['valid']
        dump(a.dataset / 'requested_balance_validation.json', balance)
        dump(a.dataset / 'finished.json', result)
        dump(a.dataset / 'progress.json', result)
        records = [json.loads(x.read_text()) for x in a.dataset.glob('scenes/*/scene_*/starts/*/configuration.json')]
        if a.allow_resampling:
            from legal_resampling import audit_replacements
            balance = audit_replacements(c,records)
            dump(a.dataset / 'requested_balance_validation.json', balance)
            dump(a.dataset / 'plan.json', [r['case'] for r in records])
            with (a.dataset / 'realized_balance.csv').open('w') as f:
                writer = csv.DictWriter(f,fieldnames=list(balance['distributions'][0]))
                writer.writeheader()
                writer.writerows({**row,'magnitude_bins':json.dumps(row['magnitude_bins'])} for row in balance['distributions'])
        counts = collections.Counter((r['case']['task'], r['case']['family'], r['case']['level'], r['accepted']) for r in records)
        audit = json.loads((a.dataset / 'physical_sampling_audit.json').read_text())
        audit['counts'] = [dict(task=t, family=f, level=l, accepted=b, count=n) for (t,f,l,b),n in sorted(counts.items())]
        contacts = collections.Counter()
        for x in searches:
            if x['status'] != 'legal_candidate_pending_official_export':
                contacts.update('|'.join(v['geoms']) for v in x['best_contacts'])
        audit.update(most_common_failed_contacts=contacts.most_common(30), contact_source='Best remaining CPU search candidates; these are diagnostic contacts, not accepted starts')
        if a.allow_resampling:
            audit['interpretation'] = 'Illegal requests were replaced by distinct legal points in the same scene/family/tier. Signed and magnitude marginals are soft targets; see realized_balance.csv. No success filtering.'
        dump(a.dataset / 'physical_sampling_audit.json', audit)
        for name in ['search_grouped_gaps.py', 'search_grouped_gaps_cpu.py', 'export_grouped_gaps_scene.py', 'publish_grouped_gaps.py', 'legal_resampling.py', 'build_grouped.py']:
            shutil.copy2(HERE / name, a.dataset / 'code' / name)
        dump(a.dataset / 'gapfill_summary.json', dict(searched=len(searches), recovered=len(verified), remaining=result['invalid'],
             previous_valid=result['valid']-len(verified), current_valid=result['valid'], unchanged_scenes_anchors_and_quotas=not a.allow_resampling,
             unchanged_scenes_anchors_families_and_tiers=True, same_level_resampling=a.allow_resampling,
             policy_actions_executed=0, previous_manifest_sha256=old_hash, manifest_sha256=new_hash))
        with (a.search / 'search_results.csv').open('w') as f:
            fields = ['task', 'scene_index', 'id', 'family', 'level', 'search_status',
                      'officially_added', 'evaluations', 'seconds', 'best_penetration_mm',
                      'actual_delta', 'best_contacts']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            added = {(json.loads((folder / 'configuration.json').read_text())['case']['task'],
                      json.loads((folder / 'configuration.json').read_text())['case']['scene_index'],
                      folder.name) for _, folder in verified}
            for x in searches:
                writer.writerow(dict(task=x['task'], scene_index=x['scene_index'], id=x['id'],
                                     family=x['case']['family'], level=x['case']['level'],
                                     search_status=x['status'], officially_added=(x['task'],x['scene_index'],x['id']) in added,
                                     evaluations=x['evaluations'], seconds=x['seconds'],
                                     best_penetration_mm=max([-v['distance_m']*1000 for v in x['best_contacts']]+[0.]),
                                     actual_delta=json.dumps(x['actual_delta']), best_contacts=json.dumps(x['best_contacts'])))
        text = (f"补点搜索完成：原来 {result['valid']-len(verified)} 个合法起点，新增 {len(verified)} 个，"
                f"现在 {result['valid']}/1460 个合法起点，仍有 {result['invalid']} 个缺口。\n"
                f"全部 {len(searches)} 个原缺口均已搜索；共进行 {sum(x['evaluations'] for x in searches):,} 次 CPU MuJoCo 几何检查。\n"+
                ("保持原场景、实际 Anchor、类别及档位，在整档重新选取合法点；允许改变原点的符号与幅度小区间，均衡作为可行域内的软目标。没有缩小档位或按任务成功筛选。\n" if a.allow_resampling else
                 "保持原场景、实际 Anchor、档位、各轴符号及幅度子区间；仅搜索区间内部幅度，没有缩小档位或按任务成功筛选。\n")+
                "新候选经过官方 detect_robot_collision、关节限位、实际扰动量、完整状态与三路 RGB 重建复核后才加入。\n"
                "原先碰撞请求常涉及底盘支撑件与柜体/柜门接触；替代点不需要沿用其原始符号。实际正负与幅度分布见 realized_balance.csv（如采用整档重采样）。\n"
                "没有把非法请求当作策略失败，也没有声称可行域内的替代点能保留不可行方向的精确均衡。\n"
                "完整逐槽位记录见 search_results.csv；旧测试集独立备份见 before_publication_backup/。\n"
                "没有加载策略、执行动作、训练或运行成功率测评。\n")
        (a.search / 'report_zh.txt').write_text(text)
        with (a.dataset / '说明.txt').open('a') as f:
            f.write('\n' + text)
        (a.dataset / 'FROZEN').write_text(new_hash + '\n')
        dump(published, dict(status='published', additions=len(verified), remaining=result['invalid'],
                             valid=result['valid'], manifest_sha256=new_hash, original_backup=str(backup)))
        print(published.read_text(), flush=True)
    except Exception:
        # Restore all originally indexed files; extra staging files are unreferenced.
        shutil.copytree(backup, a.dataset, dirs_exist_ok=True)
        dump(published, dict(status='rolled_back', original_manifest_sha256=old_hash))
        raise


if __name__ == '__main__':
    main()
