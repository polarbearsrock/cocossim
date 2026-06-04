#!/usr/bin/env python3
"""
COCOSSim model extractor using torch.fx
Traces a PyTorch nn.Module and emits a COCOSSim DAG file.

Output format (one line per compute node):
  <id> <type> <dim1> [<dim2> ...] [-> <dep_id1> <dep_id2> ...]

Supported op types:
  Matmul     M K N
  Softmax    heads seq
  LayerNorm  M hidden
  Activation total_elements

Usage:
  python extract_model.py --model llama_block --seq 1 --out graph.dag
  python extract_model.py --script my_module.py --class MyBlock --seq 512 --out graph.dag
"""

import argparse
import fnmatch
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

# --------------------------------------------------------------------------- #
# Built-in function identity helpers
# torch.matmul and torch.softmax appear as built-in methods in fx graphs;
# we match by __name__ since their id() changes across interpreter sessions.
# --------------------------------------------------------------------------- #

def _is_matmul(target) -> bool:
    return (target in (torch.matmul, torch.bmm, torch.mm)
            or getattr(target, '__name__', '') in ('matmul', 'bmm', 'mm'))

def _is_softmax(target) -> bool:
    return (target in (torch.softmax, F.softmax)
            or getattr(target, '__name__', '') == 'softmax')

def _is_activation(target) -> bool:
    return target in (F.silu, F.gelu, F.relu, F.sigmoid,
                      torch.sigmoid, torch.relu, torch.tanh)

# --------------------------------------------------------------------------- #
# Node classification
# --------------------------------------------------------------------------- #

def classify(node: torch.fx.Node, modules: Dict[str, nn.Module]) -> Optional[str]:
    """Return COCOSSim op type string, or None if this node is not a compute op."""
    if node.op == 'call_module':
        mod = modules.get(node.target)
        if isinstance(mod, nn.Linear):        return 'Matmul'
        if isinstance(mod, nn.LayerNorm):     return 'LayerNorm'
        if isinstance(mod, nn.Conv2d):        return 'Conv'
        if isinstance(mod, (nn.SiLU, nn.GELU, nn.ReLU, nn.Sigmoid)): return 'Activation'
    elif node.op == 'call_function':
        if _is_matmul(node.target):           return 'Matmul'
        if _is_softmax(node.target):          return 'Softmax'
        if _is_activation(node.target):       return 'Activation'
        if node.target is F.layer_norm:       return 'LayerNorm'
    elif node.op == 'call_method':
        if node.target in ('matmul', 'bmm', 'mm'): return 'Matmul'
        if node.target == 'softmax':               return 'Softmax'
    return None


def get_dims(node: torch.fx.Node, modules: Dict[str, nn.Module]) -> Optional[List[int]]:
    """Extract COCOSSim integer dimensions from a node's shape metadata."""
    meta = node.meta.get('tensor_meta')
    if meta is None:
        return None
    out_shape = list(meta.shape)
    op = classify(node, modules)

    if op == 'Matmul':
        if node.op == 'call_module':
            mod = modules[node.target]          # nn.Linear
            K, N = mod.in_features, mod.out_features
            M = 1
            for d in out_shape[:-1]:
                M *= d
            return [M, K, N]
        # call_function / call_method with two tensor args
        args = [a for a in node.all_input_nodes
                if a.meta.get('tensor_meta') is not None]
        if len(args) >= 2:
            a_shape = list(args[0].meta['tensor_meta'].shape)
            b_shape = list(args[1].meta['tensor_meta'].shape)
            M = 1
            for d in a_shape[:-1]:
                M *= d
            K = a_shape[-1]
            N = b_shape[-1]
            return [M, K, N]
        # fallback: read from output shape
        if len(out_shape) >= 2:
            N = out_shape[-1]
            K = out_shape[-2]
            M = 1
            for d in out_shape[:-2]:
                M *= d
            return [M, K, N]

    elif op == 'Softmax':
        # Flatten all dims except the last into "heads*batch"
        heads = 1
        for d in out_shape[:-1]:
            heads *= d
        seq = out_shape[-1]
        return [heads, seq]

    elif op == 'LayerNorm':
        if node.op == 'call_module':
            mod = modules[node.target]          # nn.LayerNorm
            norm_dims = len(mod.normalized_shape)
            M = 1
            for d in out_shape[:-norm_dims]:
                M *= d
            hidden = 1
            for d in mod.normalized_shape:
                hidden *= d
            return [M, hidden]
        # F.layer_norm: args = (input, normalized_shape, ...)
        if len(node.args) >= 2:
            norm_shape = node.args[1]           # e.g. [4096]
            norm_dims = len(norm_shape)
            M = 1
            for d in out_shape[:-norm_dims]:
                M *= d
            hidden = 1
            for d in norm_shape:
                hidden *= d
            return [M, hidden]

    elif op == 'Activation':
        total = 1
        for d in out_shape:
            total *= d
        return [total]

    elif op == 'Conv':
        if node.op == 'call_module':
            mod = modules[node.target]          # nn.Conv2d
            B = out_shape[0]
            H, W = out_shape[-2], out_shape[-1]
            C_in = mod.in_channels
            C_out = mod.out_channels
            k = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
            s = mod.stride[0]  if isinstance(mod.stride,  tuple) else mod.stride
            p = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
            return [B, C_in, H, W, C_out, k, s, p]

    return None

# --------------------------------------------------------------------------- #
# Ancestor tracking (skip through non-compute nodes)
# --------------------------------------------------------------------------- #

def compute_ancestors(node: torch.fx.Node,
                      compute_set: Set[torch.fx.Node]) -> Set[torch.fx.Node]:
    """
    Walk backward from node through non-compute intermediaries
    (reshape, transpose, add, etc.) to find the nearest compute predecessors.
    """
    result: Set[torch.fx.Node] = set()
    visited: Set[torch.fx.Node] = set()

    def walk(n: torch.fx.Node):
        if n in visited:
            return
        visited.add(n)
        for pred in n.all_input_nodes:
            if pred in compute_set:
                result.add(pred)
            else:
                walk(pred)

    for pred in node.all_input_nodes:
        if pred in compute_set:
            result.add(pred)
        else:
            walk(pred)
    return result

# --------------------------------------------------------------------------- #
# Main extraction
# --------------------------------------------------------------------------- #

def get_parallel_tag(node: torch.fx.Node, modules: Dict[str, nn.Module]) -> str:
    """
    Return the tensor-parallelism tag for a Matmul node.
      'col'  — column-parallel linear layer (nn.Linear); needs AllReduce after.
      'head' — head-parallel attention matmul (raw torch.matmul/bmm); no AllReduce.
    Non-Matmul ops return ''.
    """
    if classify(node, modules) != 'Matmul':
        return ''
    if node.op == 'call_module':
        return 'col'   # nn.Linear: output split across chiplets by column
    return 'head'      # raw matmul (attention scores / context): split by head


def extract(model: nn.Module, sample_input: torch.Tensor) -> List[dict]:
    """
    Trace model, propagate shapes, and return a list of compute node records:
      {'id': int, 'type': str, 'dims': [int,...], 'deps': [int,...], 'parallel': str}
    """
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(sample_input)
    modules = dict(model.named_modules())

    # First pass: identify all compute nodes
    compute_nodes: List[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if classify(node, modules) is not None:
            dims = get_dims(node, modules)
            if dims is not None:
                compute_nodes.append(node)

    compute_set = set(compute_nodes)
    node_to_id = {n: i for i, n in enumerate(compute_nodes)}

    # Second pass: build records with deps
    records = []
    for node in compute_nodes:
        ancestors = compute_ancestors(node, compute_set)
        records.append({
            'id':       node_to_id[node],
            'type':     classify(node, modules),
            'dims':     get_dims(node, modules),
            'deps':     sorted(node_to_id[a] for a in ancestors),
            'parallel': get_parallel_tag(node, modules),
            'fx_name':  node.name,
        })
    return records

# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

def write_dag(records: List[dict], out_path: str, comment: str = ''):
    lines = ['# COCOSSim graph v2']
    if comment:
        for line in comment.splitlines():
            lines.append(f'# {line}')
    lines.append('#')
    lines.append('# format: <id> <type>[:<parallel>] <dim1> [<dim2> ...] [-> <dep_id1> ...]')
    lines.append('# parallel tag (Matmul only): col=column-parallel linear, head=head-parallel attn')
    lines.append('')
    for r in records:
        tag      = f":{r['parallel']}" if r.get('parallel') else ''
        dims_str = ' '.join(str(d) for d in r['dims'])
        dep_str  = (' -> ' + ' '.join(str(d) for d in r['deps'])) if r['deps'] else ''
        lines.append(f"{r['id']} {r['type']}{tag} {dims_str}{dep_str}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote {len(records)} nodes to {out_path}")

# --------------------------------------------------------------------------- #
# Parallelism override
# --------------------------------------------------------------------------- #

def apply_parallel_overrides(records: List[dict], override_map: Dict[str, str]) -> None:
    """
    Apply user-supplied parallelism overrides to records in-place.

    override_map keys are matched against each record in this priority order:
      1. Node id  (e.g. "4")          — exact integer match
      2. fx_name  (e.g. "attn_q")     — exact or glob pattern (fnmatch)

    Values are the parallelism tag string: "col", "head", "row", etc.

    Example JSON:  {"attn_o": "row", "mlp_down": "row", "4": "col"}
    """
    if not override_map:
        return

    id_overrides   = {int(k): v for k, v in override_map.items() if k.lstrip('-').isdigit()}
    name_overrides = {k: v for k, v in override_map.items() if not k.lstrip('-').isdigit()}

    changed = []
    for r in records:
        new_tag = None
        # 1. id match (highest priority)
        if r['id'] in id_overrides:
            new_tag = id_overrides[r['id']]
        else:
            # 2. fx_name glob match (first pattern that matches wins)
            for pattern, tag in name_overrides.items():
                if fnmatch.fnmatch(r['fx_name'], pattern):
                    new_tag = tag
                    break
        if new_tag is not None and new_tag != r.get('parallel'):
            changed.append((r['id'], r['fx_name'], r.get('parallel', ''), new_tag))
            r['parallel'] = new_tag

    if changed:
        print("Parallelism overrides applied:")
        for node_id, fx_name, old, new in changed:
            print(f"  node {node_id:3d} [{fx_name}]: {old or '(default)'} → {new}")

# --------------------------------------------------------------------------- #
# Built-in model definitions (so the tool works without a separate model file)
# --------------------------------------------------------------------------- #

def make_llama_block(hidden: int = 4096, n_heads: int = 32,
                     ffn: int = 11008) -> nn.Module:
    """Single LLaMA-2 transformer block (attention + MLP, no embedding/unembedding)."""
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = n_heads
            self.d = hidden // n_heads
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.k = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, hidden, bias=False)
            self.o = nn.Linear(hidden, hidden, bias=False)
        def forward(self, x):
            B, T, C = x.shape
            q = self.q(x).view(B, T, self.h, self.d).transpose(1, 2)
            k = self.k(x).view(B, T, self.h, self.d).transpose(1, 2)
            v = self.v(x).view(B, T, self.h, self.d).transpose(1, 2)
            s = torch.matmul(q, k.transpose(-2, -1)) / (self.d ** 0.5)
            a = torch.softmax(s, dim=-1)
            return self.o(torch.matmul(a, v).transpose(1, 2).contiguous().view(B, T, C))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden, ffn, bias=False)
            self.up   = nn.Linear(hidden, ffn, bias=False)
            self.down = nn.Linear(ffn, hidden, bias=False)
        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.attn  = Attention()
            self.norm2 = nn.LayerNorm(hidden)
            self.mlp   = MLP()
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    return Block()


# (factory, hidden_dim)
BUILTIN_MODELS = {
    'llama7b_block':  (lambda: make_llama_block(4096, 32, 11008),  4096),
    'llama13b_block': (lambda: make_llama_block(5120, 40, 13824),  5120),
    'gpt2_block':     (lambda: make_llama_block(768,  12, 3072),    768),
}

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Extract COCOSSim DAG from a PyTorch model')
    parser.add_argument('--model',  default='llama7b_block',
                        help=f'Built-in model name: {list(BUILTIN_MODELS.keys())} '
                             f'(default: llama7b_block)')
    parser.add_argument('--script', default=None,
                        help='Path to a .py file containing a custom nn.Module subclass')
    parser.add_argument('--class',  dest='cls', default=None,
                        help='Class name to instantiate from --script')
    parser.add_argument('--seq',    type=int, default=1,
                        help='Sequence length for the sample input (default: 1 = decode)')
    parser.add_argument('--batch',  type=int, default=1,
                        help='Batch size for the sample input (default: 1)')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Hidden dimension for sample input. '
                             'Inferred automatically for built-in models; '
                             'required when using --script.')
    parser.add_argument('--out',    default='graph.dag',
                        help='Output DAG file path (default: graph.dag)')
    parser.add_argument('--show',   action='store_true',
                        help='Print the DAG to stdout as well')
    parser.add_argument('--parallel', default=None, dest='parallel_map',
                        help='JSON override map for per-node parallelism tags. '
                             'Keys: node id (e.g. "4") or fx_name glob (e.g. "attn_*"). '
                             'Values: parallelism tag ("col", "head", "row", ...). '
                             'Example: \'{"attn_o": "row", "mlp_down": "row"}\'')
    args = parser.parse_args()

    # Load model
    hidden = args.hidden
    if args.script:
        spec = importlib.util.spec_from_file_location('user_module', args.script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls  = getattr(mod, args.cls)
        model = cls()
        if hidden is None:
            print("Error: --hidden is required when using --script")
            sys.exit(1)
    else:
        if args.model not in BUILTIN_MODELS:
            print(f"Unknown model '{args.model}'. Available: {list(BUILTIN_MODELS.keys())}")
            sys.exit(1)
        factory, builtin_hidden = BUILTIN_MODELS[args.model]
        model = factory()
        if hidden is None:
            hidden = builtin_hidden

    model.eval()
    sample = torch.randn(args.batch, args.seq, hidden)

    print(f"Tracing {args.model or args.cls}  seq={args.seq}  batch={args.batch}")
    records = extract(model, sample)

    if args.parallel_map:
        try:
            override_map = json.loads(args.parallel_map)
        except json.JSONDecodeError as e:
            print(f"Error: --parallel value is not valid JSON: {e}")
            sys.exit(1)
        apply_parallel_overrides(records, override_map)

    comment = (f"model={args.model or args.cls}  seq={args.seq}  batch={args.batch}\n"
               f"nodes={len(records)}")
    write_dag(records, args.out, comment)

    if args.show:
        print()
        for r in records:
            tag      = f":{r['parallel']}" if r.get('parallel') else ''
            dims_str = ' '.join(str(d) for d in r['dims'])
            dep_str  = (' -> ' + ' '.join(str(d) for d in r['deps'])) if r['deps'] else ''
            print(f"  {r['id']:3d}  {r['type']+tag:18s}  {dims_str:30s}  {dep_str}  [{r['fx_name']}]")

if __name__ == '__main__':
    main()
