"""Run diagnosis on an existing dataset, independently of data generation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from assembly_sim.diagnosis import prepare_reference, truth_annotations
from assembly_sim.graph import build_graph,save_graph
from assembly_sim.tracing import trace_trajectory
from assembly_sim.model import POINT_NAMES
from pipeline_core import analyze_dataset,load_dataset,load_plans,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--sample',type=int,help='Optional single test index; omit to diagnose the complete test set')
    parser.add_argument('--observed-point',help='Optional final checkpoint, single-sample mode only')
    parser.add_argument('--quality-limit',type=float,default=1.0)
    parser.add_argument('--reference',type=Path,help='Optional saved normal_reference.json; otherwise use normal training samples')
    parser.add_argument('--output',type=Path,help='Output directory for full-set mode, JSON file for single-sample mode')
    args=parser.parse_args()
    try:
        if args.sample is None:
            if args.observed_point:
                raise ValueError('--observed-point requires --sample')
            output=args.output or Path(__file__).resolve().parent/'graph'/args.dataset_dir.name/('diagnosis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            result=analyze_dataset(args.dataset_dir,output,args.quality_limit,progress=print,reference_file=args.reference)
            print(json.dumps(result,ensure_ascii=False,indent=2));return 0
        values=load_dataset(args.dataset_dir);plans=load_plans(args.dataset_dir)
        if not 0<=args.sample<len(values['test']):
            raise ValueError('sample is outside the test trajectory range')
        metadata=json.loads((args.dataset_dir/'metadata.json').read_text(encoding='utf-8'))
        reference=prepare_reference(args.dataset_dir,values,plans,metadata,args.reference)
        graph=build_graph(forward=metadata.get('generation_mode')=='cad_forward',controller_policy=metadata.get('controller_policy','fixed_world'),point_names=metadata.get('point_names',POINT_NAMES))
        result=trace_trajectory(values['test'][args.sample],reference,args.quality_limit,graph=graph,
            observed_point=args.observed_point,planned=plans['test'][args.sample] if plans else None)
        result['sample_id']=f'test_{args.sample:05d}'
        output=args.output or Path(__file__).resolve().parent/'graph'/args.dataset_dir.name/f'trace_{args.sample:05d}.json'
        write_json(output,result)
        truth=truth_annotations(args.dataset_dir)
        write_json(output.parent/'ground_truth.json',truth)
        annotation=truth.get(result['sample_id'],{})
        save_graph(output.parent,graph,result['predicted_root'],result['observed_node'],
                   true_root=annotation.get('root') if annotation.get('injected') is True else None,inferred_path=result['path'])
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,TypeError,OSError,KeyError) as exc:
        print(f'Error: {exc}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
