"""Paper-ready vector figures from the current dashboard snapshot."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'figures/.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 d=json.loads((ROOT/'results/dashboard_results.json').read_text());m=d['models']['innovation']
 plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
 fig,axes=plt.subplots(2,3,figsize=(17,9));green='#278d72';blue='#5589bd'
 ax=axes[0,0];keys=['innovation','standard','generic'];values=[100*d['models'][k]['accuracy'] for k in keys]
 ax.barh(['Proposed attention','Standard attention','Generic energy'],values,color=[green,'#a6bdb2','#b1c5d2']);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('Accuracy (%)')
 for i,v in enumerate(values):ax.text(v+1,i,f'{v:.2f}%',va='center')
 ax.set_title('(a) Attention comparison',loc='left')
 ax=axes[0,1];c=np.array(m['confusion']);ax.imshow(c,cmap='Blues',vmin=0,vmax=c.max())
 for i in range(2):
  for j in range(2):ax.text(j,i,str(c[i,j]),ha='center',va='center',color='white' if i==j else '#284338',fontsize=18)
 ax.set_xticks([0,1],['Left','Right']);ax.set_yticks([0,1],['Left','Right']);ax.set_xlabel('Predicted class');ax.set_ylabel('Actual class');ax.set_title('(b) Confusion matrix',loc='left')
 ax=axes[0,2];ax.plot(m['roc']['x'],m['roc']['y'],color=green,label=f"AUC {m['auc']:.4f}");ax.plot([0,1],[0,1],'--',color='#afc1b4');ax.set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate');ax.legend(frameon=False);ax.set_title('(c) ROC curve',loc='left')
 ax=axes[1,0];ax.plot([r['epoch'] for r in m['training_loss']],[r['loss'] for r in m['training_loss']],color=green,label='Training');ax.plot([r['epoch'] for r in m['validation']],[r['loss'] for r in m['validation']],'o-',color=blue,label='Validation');ax.axvline(20,ls='--',color='#afc1b4');ax.set(xlabel='Epoch',ylabel='Cross-entropy',ylim=(0,None));ax.legend(frameon=False);ax.set_title('(d) Reference seed42 losses',loc='left')
 ax=axes[1,1];ax.plot([r['epoch'] for r in m['validation']],[100*r['accuracy'] for r in m['validation']],'o-',color=blue);ax.axvline(20,ls='--',color='#afc1b4');ax.set(xlabel='Epoch',ylabel='Validation accuracy (%)',ylim=(0,100));ax.set_title('(e) Reference seed42 validation',loc='left')
 ax=axes[1,2];parts=m['participants'];ax.bar(range(len(parts)),[100*p['accuracy'] for p in parts],color=green);ax.set_xticks(range(len(parts)),[p['id'] for p in parts],rotation=90,fontsize=6);ax.set(ylabel='Accuracy (%)',ylim=(0,100));ax.set_title('(f) All 31 validation participants',loc='left')
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
 fig.suptitle('NeuroTIA Transformer · current source-validation results',fontsize=18,fontweight='bold',color='#215c4e')
 fig.text(.5,.016,'One source fold · ensemble seeds 41/42/43 · 1,395 trials · no independent test. Learning curves: reference seed42; selected checkpoint: epoch20.',ha='center',fontsize=10,color='#687e70')
 fig.subplots_adjust(left=.09,right=.98,top=.9,bottom=.12,wspace=.4,hspace=.48)
 for ext in ['svg','pdf']:fig.savefig(ROOT/f'figures/current_source_results.{ext}',facecolor='white')
 fig2,axs=plt.subplots(1,3,figsize=(16,4.7))
 ax=axs[0];ax.plot(m['pr']['x'],m['pr']['y'],color=green);ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.set_title(f"Precision–recall · AP {m['average_precision']:.4f}",loc='left')
 ax=axs[1];ax.plot([r['confidence'] for r in m['calibration']],[r['accuracy'] for r in m['calibration']],'o-',color=green);ax.plot([0,1],[0,1],'--',color='#afc1b4');ax.set(xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1));ax.set_title('Descriptive reliability',loc='left')
 ax=axs[2];ax.bar(np.arange(10)*.05+.525,m['confidence_histogram'],width=.04,color=green);ax.set(xlabel='Maximum class probability',ylabel='Trials',xlim=(.49,1.01));ax.set_title('Prediction confidence',loc='left')
 for ax in axs:ax.spines[['top','right']].set_visible(False)
 fig2.suptitle('NeuroTIA · source-fold ensemble diagnostics',fontweight='bold',fontsize=16)
 fig2.tight_layout(rect=(0,.05,1,.9))
 for ext in ['svg','pdf']:fig2.savefig(ROOT/f'figures/current_diagnostics.{ext}',facecolor='white')
 print('Updated six-panel results and three-panel diagnostics SVG/PDF.')
if __name__=='__main__':main()
