"""Verify exported source-validation checkpoints and saved measurements."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from model.current_architecture import SignedKeyTransformer

ROOT=Path(__file__).resolve().parents[1]
def main():
    m=json.loads((ROOT/'model/current_manifest.json').read_text())
    def check(p,h): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    check('model/current_architecture.py',m['architecture_sha256'])
    check('model/architecture.py',m['base_architecture_sha256'])
    for p,h in m['artifacts_sha256'].items():check(p,h)
    scores={}
    for variant,members in m['ensemble_members'].items():
        for member in members:
            check(member['path'],member['sha256'])
            state=torch.load(ROOT/member['path'],map_location='cpu',weights_only=True)
            assert state['config']==member['config'] and state['variant']==variant
            model=SignedKeyTransformer(variant,member['config']['temperature']);model.load_state_dict(state['state_dict'])
            assert sum(p.numel() for p in model.parameters())==4920
        dashboard=json.loads((ROOT/'results/dashboard_results.json').read_text())
        row=dashboard['models'][variant]
        matrix=row['confusion'];total=sum(sum(r) for r in matrix);correct=sum(matrix[i][i] for i in range(2))
        assert total==row['trials'] and correct==row['correct']
        assert sum(p['trials'] for p in row['participants'])==total
        assert sum(p['correct'] for p in row['participants'])==correct
        scores[variant]=dict(correct=correct,trials=total)
    print(json.dumps({'scope':'Source validation only; aggregate result verification, not raw EEG replay.','scores':scores},indent=2))
if __name__=='__main__':main()
