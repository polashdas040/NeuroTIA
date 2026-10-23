"""Signed robust temporal-energy key bias; not a validated artifact detector."""
import torch
from torch.nn import functional as F
from model.architecture import MotorTokenTransformer

def variation_energy(z,variant):
    if variant=='generic':return (z-z.mean(1,keepdim=True)).square().mean(-1)
    if variant=='innovation':
        if z.shape[1]<2:raise ValueError('Require at least two latent time steps')
        differences=(z[:,1:]-z[:,:-1]).square().mean(-1)
        return torch.cat([differences[:,:1],differences],1)
    raise ValueError(variant)

def key_penalty(z,variant):
    energy=variation_energy(z,variant)
    scale=energy.median(-1,keepdim=True).values.detach().clamp_min(1e-4)
    penalty=torch.log1p(energy/scale)
    return penalty-penalty.mean(-1,keepdim=True)

class SignedKeyTransformer(MotorTokenTransformer):
    def __init__(self,variant='standard',temperature=1.):
        super().__init__(custom=False)
        if variant not in ['standard','generic','innovation'] or temperature<=0:raise ValueError('Invalid attention configuration')
        self.variant=variant;self.temperature=float(temperature)
    def forward(self,x):
        if self.variant=='standard' and self.temperature==1.:return super().forward(x)
        projected=torch.einsum('fc,nct->nft',self.spatial,x)
        filtered=F.conv1d(F.pad(projected,(31,32)),self.temporal.repeat_interleave(2,0),groups=16)
        tokens=self.local(self.first(filtered)).transpose(1,2)+self.position
        z=self.norm1(tokens);n,t,d=z.shape
        q,k,v=self.attention.qkv(z).reshape(n,t,3,4,4).permute(2,0,3,1,4)
        scores=q@k.transpose(-1,-2)/(2*self.temperature)
        if self.variant!='standard':
            penalty=key_penalty(z,self.variant)
            scores=scores-self.attention.strength.tanh()[None,:,None,None]*penalty[:,None,None,:]
        attention=scores.softmax(-1)
        update=self.attention.output((self.attention.dropout(attention)@v).transpose(1,2).reshape(n,t,d))
        tokens=tokens+self.attention_scale*update
        tokens=tokens+self.ffn_scale*self.ffn(self.norm2(tokens))
        return self.head(tokens.flatten(1)),attention
