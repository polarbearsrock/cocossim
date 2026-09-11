#!/usr/bin/env python3
"""Build the COCOSSim compute-only runner, sweep ratios, and export a report.

All build/cache data is placed under TMPDIR. Plotting uses matplotlib if installed.
The compiled runner uses the source checkout's original SA and VPU unit code.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def load_rows(path):
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def validate_results(rows):
    counts = {}
    for r in rows:
        nl = r['nonlinear_cost']
        invariant = tuple(r[k] for k in ('sa_macs', 'vpu_equiv_ops', 'sa_jobs', 'vpu_jobs'))
        if nl in counts:
            assert counts[nl] == invariant, 'Work or partitioning changed across ratios'
        counts[nl] = invariant
        assert r['mac_per_vector_op'] == r['sa_count'] * r['sa_side']**2 / (r['vpu_count'] * r['vpu_lanes'])
        for k in ('sa_occupied_pct', 'vpu_occupied_pct', 'sa_mac_util_pct', 'vpu_equiv_util_pct'):
            assert 0 <= r[k] <= 100.000001, (k, r[k])
    for nl in counts:
        rr = sorted((r for r in rows if r['nonlinear_cost'] == nl), key=lambda r: r['mac_per_vector_op'])
        # This graph/policy is expected to be monotonic; enforce it for this sweep.
        assert all(a['cycles'] <= b['cycles'] for a, b in zip(rr, rr[1:])), 'Unexpected scheduling non-monotonicity'


def report(out, rows):
    costs = sorted({int(r['nonlinear_cost']) for r in rows})
    baseline = {c: next(r for r in rows if r['nonlinear_cost'] == c and r['vpu_count'] == 72) for c in costs}
    primary = sorted((r for r in rows if r['nonlinear_cost'] == costs[0]), key=lambda r: r['mac_per_vector_op'])
    first = rows[0]
    lines = [
        '# Qwen 2.5 7B: SA / VPU compute-capacity sweep', '',
        f"One decoder layer, prefill, batch **{int(first['batch'])}**, sequence **{int(first['seq'])}** "
        f"({int(first['batch']*first['seq']):,} tokens).",
        '',
        f"Fixed **{int(first['sa_count'])} × 32×32 SAs**, varying **128-lane VPUs**, 1.9 GHz. "
        'One MAC/cell/cycle; one simple vector operation/lane/cycle. '
        'The horizontal axis is **peak MACs/cycle ÷ peak simple vector operations/cycle**. '
        'A MAC is counted once, not as two FLOPs.',
        '',
        'This is a compute-only COCOSSim extension: original SA/VPU state-machine service times, '
        'zero memory latency/traffic/contention, and a custom Qwen layer frontend with a fixed FIFO scheduler. '
        'No model weights or numerical tensors are evaluated. These are modeled timings, not measured MTIA latencies.',
        '',
        '| MAC:vector-op capacity | VPUs | Layer ms (NL=1) | Change vs 72 VPUs | SA MAC utilization | VPU utilization |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for r in primary:
        delta = 100*(r['cycles']/baseline[costs[0]]['cycles']-1)
        lines.append(f"| {r['mac_per_vector_op']:g}:1 | {int(r['vpu_count'])} | {r['time_ms']:.3f} | {delta:+.2f}% | {r['sa_mac_util_pct']:.2f}% | {r['vpu_equiv_util_pct']:.2f}% |")
    lines += ['', '## Nonlinear cost sensitivity', '',
              'NL is the lane-cycle equivalent cost of exp, reciprocal, and rsqrt; ordinary arithmetic stays at one. '
              'These are explicit throughput assumptions, not a claim about a particular SFU pipeline.', '',
              '| NL cost | 72-VPU reference | Largest tested MAC:VOP ratio within 5% of reference | VPUs | Layer ms |',
              '|---:|---:|---:|---:|---:|']
    for c in costs:
        feasible = [r for r in rows if r['nonlinear_cost'] == c and r['cycles'] <= baseline[c]['cycles']*1.05]
        best = max(feasible, key=lambda r: r['mac_per_vector_op'])
        lines.append(f"| {c} | {baseline[c]['time_ms']:.3f} ms | {best['mac_per_vector_op']:g}:1 | {int(best['vpu_count'])} | {best['time_ms']:.3f} |")
    lines += ['', '## Work and scheduling assumptions', '',
        '- Qwen dimensions: hidden 3,584; MLP 18,944; 28 query heads; 4 KV heads; head dimension 128.',
        '- Full decoder layer: two RMSNorms, Q/K/V projections and biases, RoPE, attention, output projection, '
        'two residual adds, gate/up projections, SiLU-times-up, and down projection. No embedding or LM head.',
        '- Dense masked causal attention. Both attention GEMMs execute the full S×S shape, including the masked upper triangle. '
        'Stable softmax includes scaling, causal-mask addition, max, subtract, exp, sum, reciprocal, and normalization. '
        'This is not FlashAttention or a triangular-tile-skipping schedule.',
        '- RoPE uses cached sine/cosine operands and three arithmetic operations per output element, with sign folded into subtraction. '
        'SiLU-times-up costs four ordinary operations plus exp and reciprocal per element.',
        '- Reductions retain COCOSSim’s model: one independent row per lane, with serial reduction along that row. '
        'The work census therefore counts D reduction steps for a row of length D. Parallel intra-row tree reductions are not modeled.',
        '- SA output tiles are at most 32×32 with full K accumulation and exact tail tiles. '
        'Elementwise jobs contain at most 16,384 elements; normalization and attention groups contain at most 128 rows. '
        'Job sizes and graph dependencies are identical across VPU counts.',
        '- Global FIFO ready queues for SAs and VPUs. Matrix and vector jobs may overlap when dependencies allow. '
        'Projection and MLP boundaries use operator completion; attention pipelines independently per sequence, head, and 128-row group.',
        '- Every configuration keeps all 144 SAs. This is a vector-capacity sensitivity study; area and power are not held constant.',
        '', '## Validation', '',
        '- Native SA, elementwise, and reduction tile timings checked against hand-calculated cycles.',
        '- GEMM tail tiling checked with 35×7×67 work, independently of array count.',
        '- A hand-calculated dependency chain and 50 random DAGs compare event advancement with cycle stepping.',
        '- Analytical Qwen MAC and vector-operation censuses are asserted, as are scheduled service/work totals and unit capacity.',
        '- Work and job counts are invariant across every ratio in each nonlinear-cost variant.',
        '', '## Files', '',
        '- `sweep.csv`: every ratio, timing, utilization, work count, job count, and queue statistic.',
        '- `operators.csv`: per-category service demand and first/last execution cycles. Category spans can overlap.',
        '- `timeline.csv`: time-binned SA/VPU occupancy and ready-job queue sizes.',
        '- `work_census.csv`, `native_timings.csv`: work accounting and cached native tile measurements.',
        '- `manifest.json`: commands, source revision, source hashes, and assumptions.',
        '', '## Sources', '',
        '- [Official Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)',
        '- [Qwen2 implementation, Transformers 4.43.4](https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/qwen2/modeling_qwen2.py)',
        '- [COCOSSim](https://github.com/mc186/cocossim)', '',
    ]
    (out/'report.md').write_text('\n'.join(lines))


def plots(out, rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable: CSVs and report are complete; install matplotlib to render plots.')
        return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':180})
    colors = {1:'#1565a6',4:'#d56a21'}
    costs = sorted({int(r['nonlinear_cost']) for r in rows})
    fig, ax = plt.subplots(1,2,figsize=(13,5.1),layout='constrained')
    for cost in costs:
        rr=sorted((r for r in rows if r['nonlinear_cost']==cost),key=lambda r:r['mac_per_vector_op'])
        ax[0].plot([r['mac_per_vector_op'] for r in rr],[r['time_ms'] for r in rr],
                   marker='o',markersize=4,color=colors.get(cost),label=f'Nonlinear cost = {cost}')
    primary=sorted((r for r in rows if r['nonlinear_cost']==1),key=lambda r:r['mac_per_vector_op'])
    for key,label,color in [('sa_mac_util_pct','SA MAC utilization','#1565a6'),
                             ('vpu_equiv_util_pct','VPU utilization','#d56a21')]:
        ax[1].plot([r['mac_per_vector_op'] for r in primary],[r[key] for r in primary],
                   marker='o',markersize=4,label=label,color=color)
    for a in ax:
        a.set_xscale('log',base=2)
        a.set_xticks([4,16,64,256,1024],['4:1','16:1','64:1','256:1','1024:1'])
        a.axvline(16,color='#666666',linestyle=':',linewidth=1)
        a.grid(alpha=.18)
        a.set_xlabel('Peak MAC : vector-op capacity ratio\nHigher ratio = fewer VPUs; SAs fixed')
        a.legend(frameon=False)
    ax[0].set(title='Layer latency',ylabel='Modeled latency (ms)')
    ax[1].set(title='Compute utilization (nonlinear cost = 1)',ylabel='Useful capacity utilization (%)',ylim=(0,102))
    first=rows[0]
    fig.suptitle(f"Qwen 2.5 7B · one prefill layer · BS {int(first['batch'])}, S {int(first['seq']):,}\n"
                 '144 × 32×32 SAs · 128-lane VPUs · 1.9 GHz · compute only',fontsize=14)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'sa_vpu_sweep.{ext}')
    plt.close(fig)
    tl=load_rows(out/'timeline.csv')
    chosen=[v for v in (72,9,1) if any(r['vpu_count']==v for r in primary)]
    fig,axes=plt.subplots(len(chosen),1,figsize=(12,2.4*len(chosen)),layout='constrained',squeeze=False)
    for axis,nv in zip(axes[:,0],chosen):
        rr=[r for r in tl if r['nonlinear_cost']==1 and r['vpu_count']==nv]
        for key,label,color in [('sa_occupied_pct','SA occupied','#1565a6'),('vpu_occupied_pct','VPU occupied','#d56a21')]:
            axis.plot([r['start_cycle']/1900000 for r in rr],[r[key] for r in rr],label=label,color=color,linewidth=1.3)
        result=next(r for r in primary if r['vpu_count']==nv)
        axis.set(title=f"{int(result['mac_per_vector_op'])}:1 · {nv} VPUs · {result['time_ms']:.2f} ms",
                 xlabel='Time (ms)',ylabel='Occupancy (%)',ylim=(-2,104),xlim=(0,result['time_ms']))
        axis.grid(alpha=.15);axis.legend(frameon=False,loc='upper right')
    fig.suptitle('SA/VPU overlap and idle intervals · nonlinear cost = 1',fontsize=14)
    for ext in ('png','svg'):fig.savefig(out/f'timeline.{ext}')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE.parents[1],help='extended COCOSSim source root')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--seq',type=int,default=1024)
    p.add_argument('--sa',type=int,default=144)
    p.add_argument('--vpus',default='288,144,72,36,18,9,6,4,3,2,1')
    p.add_argument('--nonlinear-costs',default='1,4')
    p.add_argument('--plot-only',action='store_true',help='render and audit existing result CSVs')
    args=p.parse_args()
    tmp=Path(os.environ['TMPDIR'])
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(tmp/'cocossim-ratio-matplotlib'))
    commands=[]
    if not args.plot_only:
        build=Path(tempfile.mkdtemp(prefix='cocossim-ratio-build-',dir=tmp))
        configure=['cmake','-S',str(HERE),'-B',str(build),f'-DCOCOSSIM_SOURCE={args.source.resolve()}',
                   '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=/usr/bin/g++']
        compile_cmd=['cmake','--build',str(build),'-j','4']
        test=[str(build/'compute_ratio'),'--selftest']
        run=[str(build/'compute_ratio'),'--batch',str(args.batch),'--seq',str(args.seq),'--sa',str(args.sa),
             '--vpus',args.vpus,'--nonlinear-costs',args.nonlinear_costs,'--out',str(out)]
        for name,cmd in [('configure',configure),('build',compile_cmd),('validation',test),('sweep',run)]:
            commands.append(cmd)
            print(name,flush=True)
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=build)
    rows=load_rows(out/'sweep.csv');validate_results(rows)
    report(out,rows);plots(out,rows)
    source=args.source.resolve()
    inputs=list((source/'include').rglob('*.h'))+[source/f'src/{f}' for f in
              ('units/standard/SysArray.cc','units/standard/VectorUnit.cc','Job.cc','global.cc','perf_enums.cc')]
    inputs += [HERE/f for f in ('runner.cc','compute_state.cc','CMakeLists.txt','run.py')]
    manifest={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'cocossim_source':str(source),
        'cocossim_revision':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        'commands':commands,
        'render_only':args.plot_only,
        'configuration':vars(args),
        'result_rows':len(rows),
        'memory_model':'absent: compute_state.cc replaces the memory-facing State implementation',
        'scheduler':'event-driven, non-preemptive, FIFO by resource type; original unit service times',
        'attention':'dense masked causal attention; no upper-triangle tile skipping',
        'ratio_definition':'SA_count * 1024 / (VPU_count * 128); MAC counts as one',
        'clock_GHz':1.9,
        'Qwen_config_url':'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json',
        'Qwen_forward_url':'https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/qwen2/modeling_qwen2.py',
    }
    if (out/'execution_record.json').exists():
        manifest['execution_record']=json.loads((out/'execution_record.json').read_text())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    print('Report:',out/'report.md',flush=True)


if __name__=='__main__':main()
