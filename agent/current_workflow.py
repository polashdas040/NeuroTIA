"""Actual local-LLM planning and reporting over verified source-study evidence."""
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
from typing import TypedDict
import numpy as np
from pydantic import BaseModel,ConfigDict,Field
from langgraph.graph import StateGraph,END
from agent.llm import call_llm,ReportReply

ROOT=Path(__file__).resolve().parents[1]
SCOPE='The available model is a frozen three-seed (41/42/43) source-fold signed temporal-energy Transformer ensemble with matched standard and generic controls. Plan evidence verification only; do not change weights or select experiments. '
class Plan(BaseModel):
    model_config=ConfigDict(extra='forbid')
    rationale:str=Field(min_length=10,max_length=1000)
    checks:list[str]=Field(min_length=1,max_length=4)
    evidence_ids:list[str]=Field(min_length=1,max_length=3)
class State(TypedDict,total=False):
    output:str
    evidence:list
    trace:list
    plan:dict
    result:dict
    report:dict
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def record(state,stage,detail):
    trace=state.get('trace',[])+[dict(stage=stage,status='completed',completed_at_utc=datetime.now(timezone.utc).isoformat(),detail=detail)]
    save(Path(state['output'])/'trace.json',trace);return trace
def retrieve():
    d=read(ROOT/'results/current_source_results.json')
    documents=[dict(id='model_protocol',source='results/current_source_results.json',text=json.dumps(dict(architecture=d['variant'],training_participants=d['training_participants'],validation_participants=d['validation_participants'],seed=d['seed'],parameters=d['parameters'],pretraining_epochs=d['pretraining_epochs'],finetuning_epochs=d['finetuning_epochs'],temperature=d['temperature'],scope=d['scope'],configurations_per_family=d['configurations_per_family'],epoch_grid=d['epoch_grid']))),
        dict(id='source_comparison',source='results/current_source_results.json',text=json.dumps({k:{x:r[x] for x in ['correct','trials','accuracy','participant_loss']} for k,r in d['metrics'].items()})),
        dict(id='evaluation_limits',source='results/current_source_results.json',text=json.dumps({k:d[k] for k in ['scope','full_cross_validation_completed','development_evaluation_performed','independent_final_test','labeled_calibration']}))]
    query='model protocol participant validation comparison accuracy evaluation limitations'
    terms=set(re.findall(r'\w+',query.lower()))
    for doc in documents:doc['retrieval_score']=len(terms & set(re.findall(r'\w+',(doc['id']+' '+doc['text']).lower())));doc['source_sha256']=sha(ROOT/doc['source'])
    return sorted(documents,key=lambda doc:(-doc['retrieval_score'],doc['id']))
def planning(state):
    docs=retrieve();save(Path(state['output'])/'retrieved_evidence.json',docs)
    reply,_=call_llm('current_planning',Plan,dict(task='Plan checks of the frozen single-fold source result and its limitations. This is planning of the present verification, not retrospective planning of training.'),docs,Path(state['output'])/'llm_calls',model_scope=SCOPE)
    return dict(evidence=docs,plan=reply,trace=record(state,'planning','Retrieved three local evidence documents; actual Ollama planning call completed.'))
def validation(state):
    m=read(ROOT/'model/current_manifest.json')
    paths={**m['artifacts_sha256'],'model/current_architecture.py':m['architecture_sha256'],'model/architecture.py':m['base_architecture_sha256']}
    paths.update({r['path']:r['sha256'] for records in m['ensemble_members'].values() for r in records})
    for path,digest in paths.items():
        if sha(ROOT/path)!=digest:raise ValueError('Integrity mismatch: '+path)
    for doc in state['evidence']:
        if sha(ROOT/doc['source'])!=doc['source_sha256']:raise ValueError('Evidence changed during run')
    return dict(trace=record(state,'validation',f'Validated {len(paths)} file hashes and retrieved evidence freshness.'))
def execution(state):
    scores={};dashboard=read(ROOT/'results/dashboard_results.json')
    for variant in ['innovation','standard','generic']:
        row=dashboard['models'][variant];matrix=row['confusion']
        total=sum(sum(r) for r in matrix);correct=sum(matrix[i][i] for i in range(2))
        assert total==row['trials'] and correct==row['correct']
        assert sum(p['trials'] for p in row['participants'])==total
        assert sum(p['correct'] for p in row['participants'])==correct
        scores[variant]=dict(correct=correct,trials=total,accuracy=correct/total,participants=len(row['participants']))
    return dict(result=dict(scores=scores,gain_percentage_points=100*(scores['innovation']['accuracy']-scores['standard']['accuracy']),scope='Aggregate source-fold confusion and participant count verification; no new EEG inference or independent test.'),trace=record(state,'execution','Python reconciled published aggregate confusion matrices and participant counts for all three models.'))

def verification(state):
    d=read(ROOT/'results/current_source_results.json')
    for variant,row in state['result']['scores'].items():
        assert row['correct']==d['metrics'][variant]['correct'] and row['trials']==d['metrics'][variant]['trials']
    assert not d['development_evaluation_performed'] and not d['independent_final_test'] and not d['full_cross_validation_completed']
    save(Path(state['output'])/'numeric_verification.json',state['result'])
    return dict(trace=record(state,'verification','Reconciled all three comparisons; confirmed one-fold scope and no independent test.'))
def reporting(state):
    reply,_=call_llm('current_reporting',ReportReply,dict(task='Report what Python verified. Clearly say one source-validation fold, settings selected on that fold, no completed full cross-validation and no new development or independent test. Do not claim the LLM improves EEG accuracy.',verification=state['result']),state['evidence'],Path(state['output'])/'llm_calls',model_scope=SCOPE)
    trace=record(state,'reporting','Actual Ollama report call completed; original prose is unverified commentary.')
    save(Path(state['output'])/'report.json',dict(numeric_verification=state['result'],planning=state['plan'],commentary=reply,commentary_verified=False,trace=trace))
    return dict(report=reply,trace=trace)
def build():
    graph=StateGraph(State);nodes=[('planning',planning),('validation',validation),('execution',execution),('verification',verification),('reporting',reporting)]
    for name,fn in nodes:graph.add_node(name,fn)
    graph.set_entry_point('planning')
    for (a,_),(b,_) in zip(nodes,nodes[1:]):graph.add_edge(a,b)
    graph.add_edge('reporting',END);return graph.compile()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);build().invoke(dict(output=str(args.output)))
    print('Actual planning and reporting completed; see report.json and llm_calls/.')
if __name__=='__main__':main()
