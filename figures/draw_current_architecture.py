"""Vector architecture diagram for NeuroTIA Transformer.

Every waveform, plate, operator, connector and mathematical label is vector.
No PNG is embedded in the SVG or PDF.
"""
from pathlib import Path
import os, json, hashlib
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle
import numpy as np

INK='#172c39'; TITLE='#005eb9'; GREEN='#d9eee3'; BLUE='#dcecff'; GOLD='#fff0ce'; PURPLE='#e4daf4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'mathtext.fontset':'dejavusans',
 'svg.fonttype':'none','pdf.fonttype':42})

def main():
 fig,ax=plt.subplots(figsize=(18,11.5))
 fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
 ax.set(xlim=(0,1800),ylim=(1150,0));ax.axis('off')
 def text(x,y,s,size=9,ha='center',weight='normal',color=INK):
  return ax.text(x,y,s,ha=ha,va='center',fontsize=size,fontweight=weight,color=color,linespacing=1.4)
 def box(x,y,w,h,s,c=BLUE,size=9,edge='#8aaed3'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=5',ec=edge,fc=c,lw=.65))
  text(x+w/2,y+h/2,s,size)
 def arrow(a,b,color=INK,style='-'):
  ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=.85,color=color,mutation_scale=9,linestyle=style,shrinkA=0,shrinkB=0))
 def wire(points,color=INK,dashed=False):
  ax.plot(*zip(*points),color=color,lw=.8,ls='--' if dashed else '-')
 def route(points):
  if len(points)>2:wire(points[:-1])
  arrow(points[-2],points[-1])
 def plus(x,y):
  ax.add_patch(Circle((x,y),11,ec=INK,fc='white',lw=.9));text(x,y,'+',12)
 def panel(x,y,w,h,title,c='#80b6ec',fill='white'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=13',ec=c,fc=fill,lw=.9,linestyle=(0,(4,3))))
  text(x+15,y+23,title,14,ha='left',weight='bold',color=TITLE)
 def plate(x,y,w,h,color):
  dx,dy=15,-17
  from matplotlib.colors import to_rgb
  rgb=np.array(to_rgb(color))
  ax.add_patch(Rectangle((x,y),w,h,fc=color,ec=rgb*.75,lw=.6))
  ax.add_patch(Polygon([(x,y),(x+dx,y+dy),(x+w+dx,y+dy),(x+w,y)],fc=np.clip(rgb*1.13,0,1),ec=rgb*.75,lw=.6))
  ax.add_patch(Polygon([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)],fc=rgb*.8,ec=rgb*.7,lw=.6))
 def tokens(x,y,w=38,h=45):
  for i in range(3):plate(x+i*9,y-i*3,w,h,'#79afea')
 def row(x,y,width,labels,c=GREEN,h=48,gap=8,size=8.3):
  w=(width-gap*(len(labels)-1))/len(labels)
  for i,s in enumerate(labels):
   box(x+i*(w+gap),y,w,h,s,c,size)
   if i:arrow((x+(i-1)*(w+gap)+w,y+h/2),(x+i*(w+gap),y+h/2))
  return w
 text(900,28,'NeuroTIA Transformer — Signed Temporal-Energy Attention',18,weight='bold',color=TITLE)
 # Complete representative member.
 panel(278,83,427,228,'CNN Spatial–Temporal Encoder','#73c8ad','#f4fbf7')
 panel(716,83,402,228,'CNN Local Feature Encoder','#8cb9e9','#f5f9ff')
 panel(1129,83,166,228,'Tokens','#8cb9e9','#f5f9ff')
 text(74,115,'EEG input\n64 × 400',10.5,weight='bold')
 t=np.linspace(0,1,140)
 for k,c in enumerate(['#438cda','#30ad9c','#d7a635','#e3786c','#8a79bb','#36aaa8']):
  # Deterministic synthetic waveform: a schematic, not measured EEG.
  z=np.sin(2*np.pi*(17+k)*t)+.45*np.sin(2*np.pi*39*t+.8*k)
  ax.plot(15+115*t,159+k*17+z*4.5,lw=.65,color=c)
 text(75,272,'⋮',16);arrow((132,202),(148,202))
 box(148,145,110,115,'Riemannian\nalignment\n'+r'$\widetilde X=R_r^{-1/2}X$',c='#e9edf3',size=9)
 arrow((258,202),(278,202))
 for x,y,w,h in [(297,151,35,101),(378,160,32,92),(463,170,29,82),(550,181,26,71)]:plate(x,y,w,h,'#3bbba9')
 text(493,134,'16 × 400',10)
 text(443,213,'⋯',16);text(530,213,'⋯',16)
 row(286,264,411,['Spatial\nprojection','Temporal\nConv k=64','BN','ELU','AvgPool\n×4','Dropout'],h=40,size=7.7)
 arrow((601,202),(731,202))
 for x,y,w,h in [(741,153,35,99),(847,168,29,84),(956,182,25,70)]:plate(x,y,w,h,'#6c9fdf')
 text(921,134,'16 × 100',10);text(915,211,'⋯',16)
 row(724,264,386,['DW Conv\nk=16','PW Conv\nk=1','BN','ELU','AvgPool\n×8','Dropout'],h=40,size=7.5)
 arrow((1002,202),(1140,202))
 plate(1148,162,29,89,'#6c9fdf');tokens(1206,191,18,55)
 text(1162,134,'16 × 12',9);text(1247,164,'12 × 16',8.5)
 box(1144,266,134,39,'Transpose + position',size=7.6)
 arrow((1271,202),(1316,202))
 box(1316,145,126,115,'Transformer\n×1\n(a)',PURPLE,11,edge='#ae94cf')
 box(1327,224,104,26,'Signed key bias (b)',GOLD,8.3,edge='#d5ac61')
 arrow((1442,202),(1467,202));plate(1472,154,22,101,'#86b5ed');text(1489,286,'Flatten\n192',9)
 arrow((1510,202),(1538,202));plate(1544,155,35,100,'#f4bd57');box(1532,267,91,39,'Linear\n192 → 2',GOLD,8.5)
 text(1705,174,'Two raw logits',9)
 route([(1604,202),(1757,202),(1757,332),(1685,332)])
 text(148,338,'4 s at 100 Hz',8.5,ha='left')
 text(148,360,'Complete unlabeled recordings',8.2,ha='left')
 text(148,381,'No labeled calibration',8.2,ha='left')
 text(148,402,'Top network: one seed member',8.2,ha='left')
 # Three independent seed pathways, aligned in a separate overall inference strip.
 panel(449,326,1294,174,'Ensemble inference — seeds 41 / 42 / 43','#e89ca9','#fffafb')
 for i,seed in enumerate([41,42,43]):
  y=364+i*40
  box(473,y,105,28,f'Seed {seed}',BLUE,8.5)
  box(610,y,226,28,'Original / mirrored logit average',BLUE,8)
  box(870,y,168,28,'0.75 mean + 0.25 median',BLUE,8.5)
  box(1074,y,108,28,'Softmax',BLUE,8.5)
  for x1,x2 in [(578,610),(836,870),(1038,1074)]:arrow((x1,y+14),(x2,y+14))
  route([(1182,y+14),(1211+i*9,y+14),(1211+i*9,405+i*10),(1262,405+i*10)])
 box(1262,388,139,64,'Mean probability\n(c)',GREEN,10)
 arrow((1401,420),(1444,420))
 text(1519,399,'Left hand',9);text(1519,443,'Right hand',9)
 box(1570,386,80,25,'','#f9c6c9',edge='#e2787c');box(1570,430,80,25,'','#bddbff',edge='#609ddf')
 text(1683,399,r'$p_L$',11);text(1683,443,r'$p_R$',11)
 text(1498,479,r'$p_i=(p_{41,i}+p_{42,i}+p_{43,i})/3$',9)
 text(469,484,'Per recording and seed: subtract 0.75 mean + 0.25 median of the binary logit difference.',7.6,ha='left')
 # Lower panels a/b/c.
 panel(15,522,573,571,'(a)  Pre-norm Transformer','#78b5ed')
 panel(603,522,574,571,'(b)  Signed temporal-energy key bias','#d8b154','#fffefa')
 panel(1192,522,593,571,'(c)  Ensemble & agent workflow','#78b5ed')
 # (a) Two residual sublayers; shape slabs at both ends.
 tokens(34,633,26,50);text(61,598,r'$Z\in\mathbb{R}^{12\times16}$',9)
 box(115,641,61,50,'LN',BLUE,10)
 box(197,641,103,50,'Signed attention',GOLD,9)
 box(325,641,60,50,r'$\gamma_A$',PURPLE,11)
 for x1,x2 in [(94,115),(176,197),(300,325),(385,424)]:arrow((x1,666),(x2,666))
 plus(435,666);arrow((446,666),(491,666));tokens(497,633,26,50)
 text(523,598,r'$Z_1$',11)
 route([(104,666),(104,723),(435,723),(435,677)])
 text(299,760,r'$Z_1=Z+\gamma_A O$',12)
 wire([(28,798),(574,798)],color='#a7b6c5',dashed=True)
 tokens(34,866,26,50);text(59,831,r'$Z_1$',10)
 # FFN components as individual compact blocks.
 xs=[111,170,252,316,392,473];ws=[43,68,50,62,68,43]
 labels=['LN','Linear\n16 → 32','GELU','Dropout','Linear\n32 → 16',r'$\gamma_F$']
 for x,w,s in zip(xs,ws,labels):box(x,877,w,43,s,BLUE if s=='LN' else PURPLE if s==r'$\gamma_F$' else GREEN,7.5)
 arrow((94,898),(111,898))
 for i in range(5):arrow((xs[i]+ws[i],898),(xs[i+1],898))
 plus(544,898);arrow((516,898),(533,898))
 route([(102,898),(102,957),(544,957),(544,909)])
 arrow((555,898),(573,898));text(559,859,r'$Z_2$',10)
 text(300,1005,r'$Z_2=Z_1+\gamma_F\operatorname{FFN}(\operatorname{LN}(Z_1))$',11)
 text(300,1052,'One encoder block • 4 heads • 4,920 parameters / member',8)
 # (b) Correct QK / innovation branches and a separate value bypass.
 text(877,580,r'$U=\operatorname{LN}(Z)\in\mathbb{R}^{12\times16}$',10.5)
 route([(877,598),(877,611),(711,611),(711,627)])
 route([(877,611),(1010,611),(1010,627)])
 box(650,627,140,55,'Q, K, V\nprojection',GOLD,9.5)
 box(640,730,160,43,r'$S_h=Q_hK_h^\top/(\sqrt{d_h}\,T)$','#fce6e6',9.3)
 arrow((720,682),(720,730))
 items=[(627,r'$E_t=\Vert U_t-U_{t-1}\Vert_2^2/16$'),(667,r'$s=\max(\operatorname{sg}(\operatorname{median}_t E_t),10^{-4})$'),(707,r'$P_t=\log(1+E_t/s)$'),(747,r'$B_{h,j}=-\tanh(\alpha_h)(P_j-\bar P)$')]
 for y,s in items:box(902,y,226,30,s,GOLD,8.8,edge='#d9b66c')
 for i in range(3):arrow((1015,items[i][0]+30),(1015,items[i+1][0]))
 plus(843,808)
 route([(720,773),(720,808),(832,808)])
 route([(1015,777),(1015,808),(854,808)])
 arrow((843,819),(843,838))
 chain=[(838,'Softmax'),(887,'Dropout'),(936,r'$\times V_h$'),(985,'Concat + Linear')]
 for y,s in chain:box(763,y,160,34,s,BLUE if y>=936 else GREEN,9)
 for i in range(3):arrow((843,chain[i][0]+34),(843,chain[i+1][0]))
 route([(650,654),(622,654),(622,953),(763,953)])
 text(635,867,r'$V_h$',9)
 arrow((843,1019),(843,1038));tokens(816,1044,25,28);text(886,1057,r'$O$',11)
 text(1035,850,r'$A_{h,ij}=\operatorname{softmax}_j(S_{h,ij}+B_{h,j})$',9.4)
 text(1035,895,r'$\beta_h=\tanh(\alpha_h)\in(-1,1)$',9)
 text(1035,930,r'$E_1=E_2,\quad T=2$',9)
 text(1035,965,'Raw values V remain unchanged',8.5)
 text(1050,1005,r'$\beta_h>0$: favor lower energy',9)
 text(1050,1050,r'$\beta_h<0$: favor higher energy',8.6)
 # (c) Three independent probability pathways and separate agent workflow.
 panel(1206,574,563,260,'Inference: multi-seed ensemble','#a3c7ed','#fbfdff')
 for i,seed in enumerate([41,42,43]):
  y=625+i*53
  cells=[(1220,78,f'Seed {seed}'),(1311,108,'Mirror logit\naverage'),(1431,103,'Centering\n0.75 mean +\n0.25 median'),(1547,72,'Softmax')]
  for x,w,s in cells:box(x,y,w,36,s,BLUE,7.2)
  for a,b in zip(cells,cells[1:]):arrow((a[0]+a[1],y+18),(b[0],y+18))
  route([(1619,y+18),(1632+i*6,y+18),(1632+i*6,687+i*10),(1655,687+i*10)])
 box(1655,676,100,61,'Mean\nprobability',GREEN,8.6)
 text(1487,809,r'$p_i=(p_{41,i}+p_{42,i}+p_{43,i})/3$',9.6)
 panel(1206,851,563,224,'LangGraph + local LLM: verification workflow','#a3c7ed','#fbfdff')
 # Evidence document is drawn from strokes, not an embedded image.
 ax.add_patch(Polygon([(1228,904),(1254,904),(1268,918),(1268,951),(1228,951)],fc='white',ec=INK,lw=.9))
 wire([(1254,904),(1254,918),(1268,918)])
 for y in [926,934,942]:wire([(1235,y),(1260,y)])
 text(1248,968,'Evidence',7.4)
 labels=['Plan','Validate','Execute','Verify','Report']
 for i,s in enumerate(labels):
  x=1295+i*90;box(x,919,78,44,s,BLUE if i in [0,4] else GREEN,7.8)
  if i:arrow((x-12,941),(x,941))
 arrow((1268,941),(1295,941))
 route([(1248,985),(1248,1000),(1694,1000),(1694,963)])
 text(1488,1041,'Verification workflow; outside EEG inference',8.5)
 # Small palette legend and explicit schematic input labeling.
 for x,c,s in [(365,GREEN,'convolution'),(625,BLUE,'tokens / feature maps'),(970,PURPLE,'Transformer'),(1240,GOLD,'proposed attention')]:
  box(x,1114,26,17,'',c,size=7);text(x+36,1123,s,8,ha='left')
 text(17,1138,'Source-fold ensemble: seeds 41/42/43. Complete-recording offline inference. Waveforms / bars schematic.',6.8,ha='left',color='#586b77')
 base=ROOT/'neurotia_signed_key_architecture'
 # Export scalable vector artifacts and a local inspection preview.
 fig.savefig(base.with_suffix('.svg'),facecolor='white')
 fig.savefig(ROOT/'neurotia_signed_key_architecture_preview.png',dpi=110,facecolor='white')
 import xml.etree.ElementTree as ET
 tree=ET.parse(base.with_suffix('.svg'))
 ns={'s':'http://www.w3.org/2000/svg'}
 assert not tree.findall('.//s:image',ns), 'Raster embedding prohibited'
 counts={name:len(tree.findall('.//s:'+name,ns)) for name in ['path','text','image']}
 manifest={'format':'true vector SVG and PDF','raster_image_elements':0,'svg_elements':counts,
 'source':'figures/draw_current_architecture.py', 'scope':'Verified 41/42/43 source-fold ensemble','figure_source':'Matplotlib vector primitives',
 'svg_sha256':hashlib.sha256(base.with_suffix('.svg').read_bytes()).hexdigest()}
 (ROOT/'signed_key_architecture_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
