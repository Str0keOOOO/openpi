"""Check final publication, preserve source starts/Anchors, and snapshot reproducible code."""
import argparse
import collections
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from grouped import dump, validate_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    a = p.parse_args()
    r, w = a.dataset, a.work
    old = w/'before_publication_backup'
    publication = json.loads((w/'publication.json').read_text())
    assert publication['status']=='published'
    config = json.loads((r/'config.json').read_text())
    manifest = json.loads((r/'frozen_manifest.json').read_text())
    validation = json.loads((r/'validation_summary.json').read_text())
    actual_hash = hashlib.sha256((r/'frozen_manifest.json').read_bytes()).hexdigest()
    assert actual_hash==(r/'FROZEN').read_text().strip()==validation['manifest_sha256']
    assert validation['passed'] and validation['valid_starts_checked']==1460
    assert manifest['planned']==manifest['valid']==len(manifest['cases'])==1460
    unchanged = 0
    for entry in json.loads((old/'frozen_manifest.json').read_text())['cases']:
        if not entry['accepted']:
            continue
        c = entry['case']
        rel = Path('scenes')/c['task']/f"scene_{c['scene_index']:02d}"/'starts'/c['id']/'configuration.json'
        assert (old/rel).read_bytes()==(r/rel).read_bytes(),str(rel)
        unchanged += 1
    anchors = 0
    for file in old.glob('scenes/*/scene_*/anchor/anchor.json'):
        for name in ['anchor.json','snapshot.npz','model.xml.gz','ep_meta.json']:
            q = file.with_name(name)
            assert q.read_bytes()==(r/q.relative_to(old)).read_bytes()
        anchors += 1
    groups = collections.Counter()
    points = collections.defaultdict(set)
    for entry in manifest['cases']:
        assert entry['accepted']
        record = json.loads(Path(entry['configuration_file']).read_text())
        case = record['case']
        assert validate_delta(config,case,record['actual_delta'])
        assert case==entry['case']
        assert entry['state_sha256']==hashlib.sha256(Path(record['state_file']).read_bytes()).hexdigest()
        assert all(Path(path).is_file() for path in record['images'].values())
        groups[(case['task'],case['scene_index'],case['family'],case['level'])] += 1
        key = (case['task'],case['scene_index'],case['family'],case['level'])
        point = tuple(np.round(np.asarray(record['actual_delta'])/config['training_limits'],9))
        assert point not in points[key]
        points[key].add(point)
    assert len(groups)==200 and all(n==(1 if key[2]=='baseline' else 8) for key,n in groups.items())
    rows = list(csv.DictReader((r/'fixed_starts.csv').open()))
    assert len(rows)==1460 and all(x['accepted']=='True' for x in rows)
    assert unchanged==1171 and anchors==20
    assert json.loads((r/'requested_balance_validation.json').read_text())['passed']
    # The default grouped config and archived code now match this actual fixed set.
    here = Path(__file__).resolve().parent
    shutil.copy2(r/'config.json',here/'grouped_config.json')
    for name in ['grouped_config.json','plot_grouped.py','audit_resampled_publication.py']:
        shutil.copy2(here/name,r/'code'/name)
    code_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').iterdir()
                   if p.is_file() and p.name!='manifest.json'}
    dump(r/'code/manifest.json',code_hashes)
    audit = dict(passed=True,legal_starts=1460,original_accepted_configs_unchanged=unchanged,
                 actual_anchors_unchanged=anchors,duplicate_points=0,
                 exact_scene_family_tier_counts=True,official_restoration_validation=True,
                 manifest_sha256=actual_hash,policy_actions_executed=0)
    dump(w/'final_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
