"""Replace illegal points anywhere in their original tier, balancing feasible marginals."""
import collections
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from grouped import digest, dump, MODEL_ARRAYS, validate_delta


def audit_replacements(config, records):
    """Hard count/tier checks, plus honest realized marginal distributions."""
    scene_counts = collections.Counter()
    groups = collections.defaultdict(list)
    for r in records:
        if not r['accepted']:
            continue
        case = r['case']
        if not validate_delta(config,case,r['actual_delta']):
            raise RuntimeError('Invalid family/tier in final replacement set')
        scene_counts[(case['task'],case['scene_index'],case['family'],case['level'])] += 1
        for axis in case['active_dimensions']:
            groups[(case['task'],case['family'],case['level'],axis)].append(r['actual_delta'][axis])
    rows = []
    exact = True
    for (task,family,level,axis), values in sorted(groups.items()):
        lo,hi = config['level_multipliers'][['small','medium','large'].index(level)]
        bins = np.minimum(4,((np.abs(values)/config['training_limits'][axis]-lo)/(hi-lo)*5).astype(int))
        positive = sum(v>0 for v in values)
        negative = sum(v<0 for v in values)
        histogram = np.bincount(bins,minlength=5).tolist()
        exact &= (axis==3 or positive==negative) and histogram==[8]*5
        rows.append(dict(task=task,family=family,level=level,axis=config['names'][axis],
                         n=len(values),positive=positive,negative=negative,magnitude_bins=histogram,
                         min=float(min(values)),max=float(max(values))))
    planned_counts = collections.Counter((r['case']['task'],r['case']['scene_index'],r['case']['family'],r['case']['level']) for r in records)
    return dict(passed=scene_counts==planned_counts, validation_scope='physical validity is verified separately; here check exact counts, family and tier',
                planned=len(records),legal_subset_count=sum(r['accepted'] for r in records),
                requested_sign_target='20 positive and 20 negative per task/family/tier/axis, except positive-only Height',
                requested_magnitude_bin_target=[8]*5, balance_policy='soft targets under physical rejection',
                exact_marginal_balance_achieved=bool(exact),legal_subset_perfectly_uniform=bool(exact),
                distributions=rows)


def describe(config, original, delta):
    case = copy.deepcopy(original)
    case['signs'] = np.sign(delta).astype(int).tolist()
    lo, hi = config['level_multipliers'][['small', 'medium', 'large'].index(case['level'])]
    case['magnitude_subbins'] = [-1] * 11
    for axis in case['active_dimensions']:
        u = (abs(delta[axis]) / config['training_limits'][axis] - lo) / (hi-lo)
        case['magnitude_subbins'][axis] = min(4, max(0, int(u*5)))
    if not validate_delta(config, case, delta):
        raise RuntimeError('Replacement left its family or tier')
    return case


def resample_missing(config, source, output, realize, contacts, limits, model, data, baseline):
    began = time.time()
    records = [json.loads(p.read_text()) for p in sorted((source/'starts').glob('*/configuration.json'))]
    accepted = [r for r in records if r['accepted']]
    chosen = []
    summaries = []
    for original in records:
        if original['accepted']:
            continue
        old = original['case']
        dest = output / old['id']
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / 'search_result.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            summaries.append(previous)
            if previous['status'] == 'legal_candidate_pending_official_export':
                chosen.append(dict(case=previous['case'], actual_delta=previous['actual_delta']))
            continue
        active = old['active_dimensions']
        lo, hi = config['level_multipliers'][['small','medium','large'].index(old['level'])]
        same = [r for r in accepted+chosen if (r['case']['family'],r['case']['level']) == (old['family'],old['level'])]
        signs = {i: collections.Counter(int(np.sign(r['actual_delta'][i])) for r in same) for i in active}
        bins = {i: collections.Counter(describe(config,r['case'],np.asarray(r['actual_delta']))['magnitude_subbins'][i] for r in same) for i in active}
        # Soft targets preserve coverage when possible, without forcing collisions.
        targets = {i: collections.Counter(r['case']['magnitude_subbins'][i] for r in records
                    if (r['case']['family'],r['case']['level']) == (old['family'],old['level'])) for i in active}
        existing = {tuple(np.round(np.asarray(r['actual_delta'])/config['training_limits'],9)) for r in same}
        seed = int(digest([config['seed'],old,'whole_tier_replacement'])[:8],16)
        rng = np.random.default_rng(seed)
        sampler = qmc.Sobol(d=len(active)*2, scramble=True, seed=seed)
        points = sampler.random_base2(12)
        best = None
        legal_count = 0
        evaluations = 0
        rejected_contacts = collections.Counter()
        start = time.time()
        for n, point in enumerate(points):
            delta = np.zeros(11)
            for j, axis in enumerate(active):
                lower = max(lo*config['training_limits'][axis], np.nextafter(config['tolerances'][axis], np.inf))
                upper = hi*config['training_limits'][axis]
                sign = 1 if axis == 3 or point[j+len(active)] >= .5 else -1
                delta[axis] = sign*(lower + point[j]*(upper-lower))
            actual = realize(delta)
            coll = contacts()
            evaluations += 1
            if coll or limits():
                rejected_contacts.update('|'.join([model.geom(int(x.geom1)).name or '',model.geom(int(x.geom2)).name or '']) for x in coll)
            elif tuple(np.round(actual/config['training_limits'],9)) not in existing:
                case = describe(config,old,actual)
                score = float(rng.uniform(0,.001))
                for axis in active:
                    s = case['signs'][axis]
                    b = case['magnitude_subbins'][axis]
                    if axis != 3:
                        score += 2*(2*(signs[axis][s]-4)+1)
                    score += 2*(bins[axis][b]-targets[axis][b])+1
                legal_count += 1
                if best is None or score < best[0]:
                    best = (score, actual.copy(), case)
            # Consider a broad pool; expand automatically if the feasible set is small.
            if n >= 511 and legal_count >= 32:
                break
        if best is None:
            result = dict(case=old, original_case=old, task=old['task'],scene_index=old['scene_index'],id=old['id'],
                          status='not_found_in_whole_tier',evaluations=evaluations,seconds=time.time()-start,
                          actual_delta=original['actual_delta'],best_contacts=[],most_common_rejected_contacts=rejected_contacts.most_common(10))
        else:
            _, actual, case = best
            realize(actual)
            if contacts() or limits():
                raise RuntimeError('Selected replacement is not reproducibly legal')
            state = np.r_[data.time, data.qpos, data.qvel]
            if len(state) != len(baseline['state']):
                raise RuntimeError('Unexpected state activation layout')
            np.savez_compressed(dest/'candidate_state.npz',state=state,qpos=data.qpos.copy(),qvel=data.qvel.copy(),
                                ctrl=data.ctrl.copy(),**{key:np.asarray(getattr(model,key)).copy() for key in MODEL_ARRAYS})
            result = dict(case=case,original_case=old,task=old['task'],scene_index=old['scene_index'],id=old['id'],
                          status='legal_candidate_pending_official_export',evaluations=evaluations,seconds=time.time()-start,
                          actual_delta=actual.tolist(),best_contacts=[],feasible_pool_size=legal_count,
                          original_anchor_unchanged=True,original_scene_and_tier_unchanged=True,
                          sampling_mode='same_level_legal_resampling',original_tier_sign_subbin_unchanged=False,
                          most_common_rejected_contacts=rejected_contacts.most_common(10))
            chosen.append(dict(case=case, actual_delta=actual.tolist()))
        result['policy_actions_executed'] = 0
        dump(marker,result)
        summaries.append(result)
        print(old['task'],old['scene_index'],old['id'],result['status'],evaluations,flush=True)
    dump(output/'summary.json',dict(task=records[0]['case']['task'],scene_index=records[0]['case']['scene_index'],
         missing=len(summaries),found=sum(x['status']=='legal_candidate_pending_official_export' for x in summaries),
         not_found=sum(x['status']!='legal_candidate_pending_official_export' for x in summaries),
         seconds=time.time()-began,policy_actions_executed=0))
