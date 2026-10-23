"""Shared offline inference primitives and current-model command entry point."""
import hashlib,re
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
 digest=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
 return digest.hexdigest()

def reflection(channels):
 names=[str(c).rstrip('.').upper() for c in channels];lookup={c:i for i,c in enumerate(names)};result=[]
 for name in names:
  match=re.fullmatch(r'([A-Z]+)(\d+)',name)
  partner=match.group(1)+str(int(match.group(2))+(1 if int(match.group(2))%2 else -1)) if match else name
  result.append(lookup[partner])
 result=np.asarray(result,dtype=np.int64)
 if not np.array_equal(result[result],np.arange(len(names))):raise ValueError('Invalid channel reflection')
 return result

def predict_member(model,x,permutation,recordings,centering_mix=0):
 if not 0<=centering_mix<=1:raise ValueError("Invalid frozen centering mixture")
 model.eval();parts=[]
 with torch.no_grad():
  for start in range(0,len(x),64):
   batch=x[start:start+64]
   logits=.5*(model(batch)[0]+model(batch[:,permutation])[0][:,[1,0]])
   parts.append(logits.numpy())
 logits=np.concatenate(parts)
 for recording in np.unique(recordings):
  mask=recordings==recording;logits[mask]-=logits[mask].mean(0,keepdims=True)
 if centering_mix:
  for recording in np.unique(recordings):
   mask=recordings==recording
   bias=.5*centering_mix*np.median(logits[mask,1]-logits[mask,0])
   logits[mask,1]-=bias;logits[mask,0]+=bias
 return torch.from_numpy(logits).softmax(-1).numpy()

if __name__=="__main__":
 from model.predict_current import main
 main()
