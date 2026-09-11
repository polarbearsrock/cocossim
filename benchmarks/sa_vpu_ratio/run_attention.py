#!/usr/bin/env python3
"""Run and explain Qwen2.5-7B prefill attention on two SAs and one VPU."""
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
CYCLES_PER_MS = 1_900_000
SOURCES = {
    'Qwen configuration': 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json',
    'Reference attention implementation': 'https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/qwen2/modeling_qwen2.py',
}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def integer(row, key):
    return int(row[key])


def audit(rows, stages):
    allowed = {'RMSNorm', 'QKV', 'QKV_bias', 'RoPE', 'Attention_QK', 'Softmax', 'Attention_AV', 'O_projection', 'Residual'}
    for result in rows:
        nl = integer(result, 'nonlinear_cost')
        batch, seq = integer(result, 'batch'), integer(result, 'seq')
        assert (integer(result, 'sa_count'), integer(result, 'vpu_count')) == (2, 1)
        assert float(result['mac_per_vector_op']) == 16
        expected_mac = batch*seq*(2*3584**2 + 2*3584*4*128) + 2*batch*28*seq**2*128
        expected_vop = batch*seq*(4*3584+2+nl + 3584+2*4*128 + 3*(3584+4*128) + 3584)
        expected_vop += batch*28*seq*((6+nl)*seq+nl)
        assert integer(result, 'sa_macs') == expected_mac
        assert integer(result, 'vpu_equiv_ops') == expected_vop
        ss = {integer(s, 'stage_id'): s for s in stages if integer(s, 'nonlinear_cost') == nl}
        assert set(s['category'] for s in ss.values()) == allowed
        for s in ss.values():
            ready, first, last = (integer(s, k) for k in ('ready_cycle', 'first_cycle', 'last_cycle'))
            assert ready <= first < last <= integer(result, 'cycles')
            for dep in filter(None, s['dependencies'].split(';')):
                assert integer(ss[int(dep)], 'last_cycle') <= ready
            capacity = 2 if s['resource'] == 'SA' else 1
            assert integer(s, 'service_cycles') <= capacity*(last-first)
        for resource, prefix, work_key in [('SA', 'sa', 'sa_macs'), ('VPU', 'vpu', 'vpu_equiv_ops')]:
            selected = [s for s in ss.values() if s['resource'] == resource]
            assert sum(integer(s, 'work') for s in selected) == integer(result, work_key)
            assert sum(integer(s, 'jobs') for s in selected) == integer(result, prefix+'_jobs')


def render(out):
    rows, stages, ops = (read_csv(out/name) for name in ('sweep.csv', 'stages.csv', 'operators.csv'))
    audit(rows, stages)
    first = rows[0]
    batch, seq = integer(first, 'batch'), integer(first, 'seq')
    primary_nl = min(integer(r, 'nonlinear_cost') for r in rows)
    primary = next(r for r in rows if integer(r, 'nonlinear_cost') == primary_nl)
    cats = {r['category']: r for r in ops if integer(r, 'nonlinear_cost') == primary_nl}
    ss = {s['name']: s for s in stages if integer(s, 'nonlinear_cost') == primary_nl}
    core_ms = (integer(cats['Attention_AV'], 'last_cycle')-integer(cats['Attention_QK'], 'first_cycle'))/CYCLES_PER_MS
    qk_cycles = integer(ss['score_0_0_0'], 'service_cycles')/2
    av_cycles = integer(ss['av_0_0_0'], 'service_cycles')/2
    lines = [
        '# Qwen2.5-7B attention: two 32×32 SAs and one 128-lane VPU', '',
        f'Prefill, batch **{batch}**, sequence **{seq:,}**, **{batch*seq:,} tokens**, one self-attention sublayer. '
        'This includes input RMSNorm and the output residual add; the MLP is excluded.', '',
        '**Compute only:** original COCOSSim SA/VPU state machines with immediate memory completion, '
        'a custom operator DAG, and FIFO scheduling. There are no memory, network, or other compute resources. '
        'This predicts timing without executing numerical tensors or validating a particular chip.', '',
        'Peak provisioning is **2,048 MACs/cycle : 128 simple vector ops/cycle = 16:1**. '
        'One MAC is counted once. SAs issue one MAC per cell per cycle; the VPU issues one ordinary op per lane per cycle. '
        'The clock assumption is 1.9 GHz.', '',
        '## Result', '',
        '| Nonlinear lane-cycle cost | Cycles | Sublayer latency (ms) | SA occupied | SA useful MAC utilization | VPU occupied |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['nonlinear_cost']} | {int(r['cycles']):,} | {float(r['time_ms']):.6f} | {float(r['sa_occupied_pct']):.2f}% | {float(r['sa_mac_util_pct']):.2f}% | {float(r['vpu_occupied_pct']):.2f}% |")
    lines += ['', 'Nonlinear cost assigns exp, reciprocal, and rsqrt either one or four lane-cycle equivalents. '
              'This is a throughput sensitivity assumption, not a measured SFU instruction latency.', '',
              f'The central QK → softmax → AV interval takes **{core_ms:.3f} ms**. '
              'QKV and output projections account for most of the remaining time.', '',
              '## How the layer maps', '',
              f'The input is [{batch}, {seq}, 3584], flattened to [{batch*seq}, 3584] for projections. '
              'There are 28 query heads, four K/V heads, and 128 elements per head. Seven query heads share each K/V head.', '',
              '| Operation | Resource | Matrix shape or vector work |', '|---|---|---|',
              '| Input RMSNorm | VPU | One normalization over 3584 values per token |',
              f'| Q projection | Both SAs | [{batch*seq},3584] × [3584,3584] |',
              f'| K and V projections | Both SAs | Each [{batch*seq},3584] × [3584,512] |',
              '| Bias and RoPE | VPU | Bias on Q/K/V; cached-table rotary arithmetic on Q/K |',
              f'| QKᵀ | Both SAs | Per sequence/query head: [{seq},128] × [128,{seq}] |',
              '| Scale, mask, stable softmax | VPU | Max and sum reductions, exp, reciprocal, normalization |',
              f'| AV | Both SAs | Per sequence/query head: [{seq},{seq}] × [{seq},128] |',
              f'| Output projection | Both SAs | [{batch*seq},3584] × [3584,3584] |',
              '| Residual add | VPU | One add per output element |', '',
              'Every GEMM is split into output tiles of at most 32×32 with full K accumulation. '
              'The two SAs execute two independent tiles at once. Attention has independent groups of up to 128 query rows. '
              'The VPU reduces each row serially, with up to 128 independent rows in parallel.', '',
              '## Scheduled timeline', '',
              'Times below describe category execution windows. Vector windows include gaps and can overlap SA windows.', '',
              '| Operation | Starts (ms) | Ends (ms) | Active service / available units (ms) |',
              '|---|---:|---:|---:|']
    for category in ('RMSNorm','QKV','QKV_bias','RoPE','Attention_QK','Softmax','Attention_AV','O_projection','Residual'):
        c = cats[category]
        service = (integer(c, 'sa_busy_cycles')/2 + integer(c, 'vpu_busy_cycles'))/CYCLES_PER_MS
        lines.append(f"| {category} | {int(c['first_cycle'])/CYCLES_PER_MS:.3f} | {int(c['last_cycle'])/CYCLES_PER_MS:.3f} | {service:.3f} |")
    lines += ['', 'The FIFO queues dispatch Q, K, then V projections. Bias and rotary work overlap later projections. '
              'All QK groups enter the SA queue before AV jobs become ready, so SAs complete the QK groups, then the AV groups. '
              'Softmax overlaps QK execution. This is the modeled scheduling policy; a scheduler that prioritizes ready AV groups would produce a different timeline.', '',
              'The plot averages occupancy over 1,000,000-cycle bins (about 0.526 ms at 1.9 GHz).', '',
              '![SA and VPU timeline](attention_timeline.png)', '',
              '## One attention group', '',
              f'For the first group ({min(128,seq)} query rows, one query head, one sequence):', '',
              '| Operation | Jobs | Service cycles with these resources | Time (µs) |',
              '|---|---:|---:|---:|',
              f"| QKᵀ | {ss['score_0_0_0']['jobs']} SA tiles | {qk_cycles:,.0f} | {qk_cycles/1900:.3f} |"]
    for r in rows:
        cost = r['nonlinear_cost']
        sm = next(s for s in stages if s['name']=='softmax_0_0_0' and s['nonlinear_cost']==cost)
        cycles = integer(sm, 'service_cycles')
        lines.append(f'| Softmax, nonlinear cost {cost} | 1 VPU group | {cycles:,} | {cycles/1900:.3f} |')
    lines += [f"| AV | {ss['av_0_0_0']['jobs']} SA tiles | {av_cycles:,.0f} | {av_cycles/1900:.3f} |", '',
              'These are service times without queue waiting. A group depends on its own QK → softmax → AV chain; '
              'other groups can overlap it. For S=1024, the two SAs produce a group of scores every 10,304 cycles. '
              'Softmax consumes 7,175 cycles at nonlinear cost 1 or 10,250 at cost 4: about 69.6% or 99.5% of that production interval. '
              '**Low average VPU occupancy does not imply generous headroom during softmax.**', '',
              '## Boundaries and validation', '',
              '- Dense masked causal attention: both GEMMs execute the full square, including the masked upper triangle. No FlashAttention or causal tile skipping.',
              '- Cached RoPE tables; no table generation, data movement, packing, or dtype conversion cost.',
              '- Projection boundaries wait for the whole projection. Attention dependencies are per 128-row group. SRAM capacity is unconstrained because memory is excluded.',
              '- Native tile service, clock isolation, exact tail work, attention scope, and a two-SA/one-VPU dependency chain are checked. Fifty deterministic random DAGs compare event and cycle stepping.',
              '- Every run checks work/service conservation, resource capacity, and producer completion. This report independently audits all per-stage dependencies and analytical attention MAC/vector totals.',
              f"- Total work: {int(primary['sa_macs']):,} MACs; {int(primary['vpu_equiv_ops']):,} vector equivalents for nonlinear cost {primary_nl}.", '',
              '## Timing correction', '',
              'The earlier sweep initialized a native SA before resetting the oracle clock. Native SA initialization captures '
              'the dispatch timestamp, so shorter following measurements could acquire an artificial completion stall. '
              'The reset now happens before initialization. At 32×32 output, K=128 takes 161 cycles and K=1024 takes 1057 cycles; '
              'the old sweep recorded 385 and 2049. This report uses corrected timings. The prior full-layer sweep was also rerun.', '',
              '## Sources', '']
    lines += [f'- [{name}]({url})' for name,url in SOURCES.items()]
    (out/'report.md').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    timeline = read_csv(out/'timeline.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 3.0*len(rows)), layout='constrained', squeeze=False)
    for axis,r in zip(axes[:,0], rows):
        points = [t for t in timeline if t['nonlinear_cost']==r['nonlinear_cost']]
        x = [(int(t['start_cycle'])+int(t['duration'])/2)/CYCLES_PER_MS for t in points]
        for key,label,color in [('sa_occupied_pct','SA occupancy (both arrays)','#1565a6'),('vpu_occupied_pct','VPU occupancy','#d56a21')]:
            axis.plot(x,[float(t[key]) for t in points],color=color,linewidth=1.25,label=label)
        for cat,label in [('QKV','QKV projections'),('Attention_QK','QKᵀ'),('Attention_AV','AV'),('O_projection','Output projection')]:
            c = next(c for c in ops if c['category']==cat and c['nonlinear_cost']==r['nonlinear_cost'])
            start,end = (int(c[k])/CYCLES_PER_MS for k in ('first_cycle','last_cycle'))
            axis.axvspan(start,end,color='#1565a6',alpha=.035)
            axis.text((start+end)/2,108,label,ha='center',va='bottom',fontsize=9)
        axis.set(xlim=(0,float(r['time_ms'])),ylim=(-3,125),yticks=[0,25,50,75,100],xlabel='Time (ms)',ylabel='Occupancy (%)',
                 title=f"Nonlinear cost {r['nonlinear_cost']} · total {float(r['time_ms']):.3f} ms · average VPU {float(r['vpu_occupied_pct']):.2f}%")
        axis.grid(axis='y',alpha=.16)
        axis.legend(loc='center right',frameon=False)
    fig.suptitle(f'Qwen2.5-7B prefill attention · BS {batch}, S {seq:,}\n2 × 32×32 SAs + 1 × 128-lane VPU · 1.9 GHz · compute only',fontsize=14)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/f'attention_timeline.{ext}')
    plt.close(fig)
    print('PASS: attention work census, scope, trace dependencies, and capacity audits')
    print('Report:',out/'report.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE.parents[1])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--seq',type=int,default=1024)
    p.add_argument('--nonlinear-costs',default='1,4')
    p.add_argument('--binary',type=Path,help='Use an already built runner')
    p.add_argument('--plot-only',action='store_true')
    args = p.parse_args()
    tmp = Path(os.environ['TMPDIR'])
    os.environ.setdefault('MPLCONFIGDIR',str(tmp/'cocossim-ratio-matplotlib'))
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    previous_manifest = json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
    commands = []
    def execute(name, cmd):
        commands.append(cmd)
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    if not args.plot_only:
        binary = args.binary
        if binary is None:
            build = Path(tempfile.mkdtemp(prefix='cocossim-attention-build-',dir=tmp))
            execute('configure',['cmake','-S',str(HERE),'-B',str(build),f'-DCOCOSSIM_SOURCE={args.source.resolve()}',
                                 '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=/usr/bin/g++'])
            execute('build',['cmake','--build',str(build),'-j','4'])
            binary = build/'compute_ratio'
        execute('validation',[str(binary),'--selftest'])
        execute('attention',[str(binary),'--scope','attention','--batch',str(args.batch),'--seq',str(args.seq),
                             '--sa','2','--vpus','1','--nonlinear-costs',args.nonlinear_costs,
                             '--bin-width','1000000','--trace-stages','1','--out',str(out)])
    render(out)
    source = args.source.resolve()
    inputs = list((source/'include').rglob('*.h'))+[source/f'src/{f}' for f in
              ('units/standard/SysArray.cc','units/standard/VectorUnit.cc','Job.cc','global.cc','perf_enums.cc')]
    inputs += [HERE/f for f in ('runner.cc','compute_state.cc','CMakeLists.txt','run_attention.py')]
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),'configuration':vars(args),
                'commands':commands or previous_manifest.get('commands',[]),
                'last_action':'render existing results' if args.plot_only else 'run and render',
                'cocossim_revision':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
                'scope':'input RMSNorm through self-attention residual; excludes MLP',
                'timing_correction':'oracle clock reset precedes native init',
                'clock_GHz':1.9,'memory':'absent','attention':'dense masked causal',
                'scheduler':'FIFO by resource type; whole-projection and per-attention-group dependencies','sources':SOURCES}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')


if __name__=='__main__':
    main()
