"""Prepare complete PhysioNet imagery recordings for model.predict (no labels)."""
import argparse,json,re
from pathlib import Path
import mne
import numpy as np
from model.predict import ROOT

def prepare(paths,output):
 channels=json.loads((ROOT/'model/channels.json').read_text());parts=[];ids=[];recordings=[]
 for path in paths:
  match=re.fullmatch(r'(S\d{3}R(?:04|08|12))\.edf',path.name,re.IGNORECASE)
  if not match:raise ValueError('Require PhysioNet imagery runs 04, 08 or 12')
  recording=match.group(1).upper()
  if recording in recordings:raise ValueError('Duplicate recording')
  with mne.io.read_raw_edf(path,preload=True,verbose='ERROR') as raw:
   if raw.ch_names!=channels:raise ValueError('Channel order mismatch')
   seconds=raw.n_times/raw.info['sfreq'];filtered=raw.copy().notch_filter(50,verbose='ERROR')
   filtered.filter(4,38,phase='zero',verbose='ERROR');filtered.set_eeg_reference('average',projection=False,verbose='ERROR')
   filtered.resample(100,verbose='ERROR');values=filtered.get_data()*1e6;local=[];localids=[];ann=raw.annotations
   for index,(absolute,duration,label) in enumerate(zip(ann.onset,ann.duration,ann.description)):
    if label not in ('T1','T2'):continue
    onset=float(absolute-raw.first_time);next_onset=float(ann.onset[index+1]-raw.first_time) if index+1<len(ann) else seconds
    start=int(filtered.time_as_index(onset,use_rounding=True)[0]);epoch=values[:,start:start+400]
    if onset<0 or onset+4>min(onset+duration,next_onset,seconds)+1e-8 or epoch.shape!=(64,400) or not np.isfinite(epoch).all():continue
    local.append(epoch-epoch.mean(axis=-1,keepdims=True));localids.append(f'{recording}A{index:03d}')
   if len(local)<2:raise ValueError('Recording has fewer than two eligible trials')
   local=np.stack(local);gain=float(np.sqrt(np.mean(local**2)))
   if gain<=0:raise ValueError('Zero signal RMS')
   parts.append((local/gain).astype(np.float32));ids.extend(localids);recordings.extend([recording]*len(local))
 if output.suffix.lower()!='.npz':raise ValueError('Output must be .npz')
 with output.open('xb') as f:np.savez_compressed(f,x=np.concatenate(parts),trial_id=np.array(ids),recording_id=np.array(recordings),channels=np.array(channels))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('recordings',nargs='+',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();prepare(args.recordings,args.output)
