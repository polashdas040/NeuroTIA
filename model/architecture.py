"""Learned motor-power token Transformer with candidate innovation attention."""
import torch
from torch import nn
from torch.nn import functional as F


class InnovationAttention(nn.Module):
    def __init__(self,custom):
        super().__init__();self.custom=custom
        self.qkv=nn.Linear(16,48);self.output=nn.Linear(16,16)
        self.strength=nn.Parameter(torch.full((4,),-1.))
        self.dropout=nn.Dropout(.15)
    def forward(self,x):
        n,t,d=x.shape
        q,k,v=self.qkv(x).reshape(n,t,3,4,4).permute(2,0,3,1,4)
        scores=q@k.transpose(-1,-2)/2
        if self.custom:
            # Agreement of changes in learned sensorimotor features across time.
            delta=torch.cat([torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]],1)
            delta=F.normalize(delta,dim=-1)
            agreement=delta@delta.transpose(-1,-2)
            scores=scores+F.softplus(self.strength)[None,:,None,None]*agreement[:,None]
        attention=scores.softmax(-1)
        mixed=(self.dropout(attention)@v).transpose(1,2).reshape(n,t,d)
        return self.output(mixed),attention


class MotorTokenTransformer(nn.Module):
    def __init__(self,custom=True):
        super().__init__()
        self.spatial=nn.Parameter(torch.randn(16,64)*.04)
        self.temporal=nn.Parameter(torch.randn(8,1,64)*.03)
        self.first=nn.Sequential(nn.BatchNorm1d(16,eps=.001),nn.ELU(),nn.AvgPool1d(4),nn.Dropout(.25))
        self.local=nn.Sequential(nn.Conv1d(16,16,16,padding='same',groups=16,bias=False),nn.Conv1d(16,16,1,bias=False),
            nn.BatchNorm1d(16,eps=.001),nn.ELU(),nn.AvgPool1d(8),nn.Dropout(.25))
        self.position=nn.Parameter(torch.randn(1,12,16)*.01)
        self.norm1=nn.LayerNorm(16);self.norm2=nn.LayerNorm(16)
        self.attention=InnovationAttention(custom)
        self.ffn=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Dropout(.25),nn.Linear(32,16))
        self.head=nn.Linear(12*16,2)
        self.attention_scale=nn.Parameter(torch.tensor(.1));self.ffn_scale=nn.Parameter(torch.tensor(.1))
    def forward(self,x):
        projected=torch.einsum('fc,nct->nft',self.spatial,x)
        temporal=self.temporal.repeat_interleave(2,0)
        filtered=F.conv1d(F.pad(projected,(31,32)),temporal,groups=16)
        tokens=self.local(self.first(filtered)).transpose(1,2)+self.position
        update,attention=self.attention(self.norm1(tokens))
        tokens=tokens+self.attention_scale*update
        tokens=tokens+self.ffn_scale*self.ffn(self.norm2(tokens))
        return self.head(tokens.flatten(1)),attention
    @torch.no_grad()
    def constrain(self):
        self.spatial.mul_((1/self.spatial.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp(max=1))
        self.head.weight.mul_((.5/self.head.weight.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp(max=1))
