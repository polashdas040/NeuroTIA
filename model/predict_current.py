"""Offline inference with the 41/42/43 source-fold ensemble."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from model.current_architecture import SignedKeyTransformer
from model.alignment import align_recording
from model.predict import reflection,predict_member,sha256

ROOT=Path(__file__).resolve().parents[1]
def classify(x,channels,recordings,variant='innovation'):
    manifest=json.loads((ROOT/'model/current_manifest.json').read_text())
    members=manifest['ensemble_members'][variant]
    for path,digest in {**{m['path']:m['sha256'] for m in members},'model/current_architecture.py':manifest['architecture_sha256'],'model/architecture.py':manifest['base_architecture_sha256'],'model/predict.py':manifest['shared_predictor_sha256']}.items():
        if sha256(ROOT/path)!=digest:raise ValueError('Current model integrity mismatch: '+path)
    if list(channels)!=json.loads((ROOT/'model/channels.json').read_text()):raise ValueError('Require saved channel order')
    x=np.asarray(x,dtype=np.float32).copy();recordings=np.asarray(recordings)
    if x.ndim!=3 or x.shape[1:]!=(64,400) or not len(x) or recordings.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError('Require finite N × 64 × 400 EEG and N recording IDs')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for recording in np.unique(recordings):
            mask=recordings==recording
            if mask.sum()<2:raise ValueError('Complete multi-trial recording batches required')
            x[mask]=align_recording(x[mask])[0]
    predictions=[]
    for member in members:
        state=torch.load(ROOT/member['path'],map_location='cpu',weights_only=True)
        if state['config']!=member['config'] or state['variant']!=variant or state['seed']!=member['seed']:raise ValueError('Checkpoint configuration mismatch')
        model=SignedKeyTransformer(variant,member['config']['temperature']);model.load_state_dict(state['state_dict'])
        predictions.append(predict_member(model,torch.from_numpy(x),reflection(channels),recordings,.25))
    return np.stack(predictions).mean(0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--variant',choices=['innovation','standard','generic'],default='innovation');p.add_argument('--complete-recordings',action='store_true');a=p.parse_args()
    if not a.complete_recordings:raise ValueError('Complete offline recording batches must be confirmed')
    if a.output.exists() or a.output.suffix.lower()!='.npz':raise ValueError('Output must be a new NPZ file')
    with np.load(a.input,allow_pickle=False) as z:x=z['x'];channels=z['channels'];recordings=z['recording_id'];ids=z['trial_id']
    if len(ids)!=len(x) or len(np.unique(ids))!=len(ids):raise ValueError('Require unique trial IDs')
    prob=classify(x,channels,recordings,a.variant)
    with a.output.open('xb') as f:np.savez_compressed(f,probabilities=prob,predicted_class=prob.argmax(1),trial_id=ids,recording_id=recordings,variant=a.variant,seeds=np.array([41,42,43]),scope='source-fold ensemble')
    print(f'Saved {len(prob)} predictions from the {a.variant} source-fold ensemble.')
if __name__=='__main__':main()
