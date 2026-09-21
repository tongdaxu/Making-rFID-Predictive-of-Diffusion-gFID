from pathlib import Path
import re

p = Path("evalvae_fix_recon_topkmean.py")
s = p.read_text()

# 1. Add argument.
if "--top-agg" not in s:
    anchor = '    parser.add_argument("--top", type=int, default=10)\n'
    if anchor not in s:
        raise RuntimeError('Cannot find parser anchor: parser.add_argument("--top", ...). Please grep --top manually.')
    s = s.replace(
        anchor,
        anchor + '    parser.add_argument("--top-agg", type=str, default="sample", choices=["sample", "mean", "nearest"], help="How to aggregate top-k NN candidates: sample/original stochastic choice, mean average, or nearest top-1.")\n'
    )

# 2. Patch the common categorical / multinomial selection block.
# This is deliberately pattern-based, because the exact variable names in this file matter.
# We search for the place after top-k candidates are prepared and before interpolation uses znn.

patterns = [
    # Pattern A: torch.multinomial based sampling.
    (
        r'(?P<block>'
        r'(?P<indent>[ \t]*)prob\s*=\s*torch\.softmax\([^\n]+\)\n'
        r'(?P=indent).*?torch\.multinomial\([^\n]+\)\.squeeze[^\n]*\n'
        r'(?P=indent)znn\s*=\s*znn_candidate\[[^\n]+\]\n'
        r')',
        'multinomial'
    ),

    # Pattern B: Categorical(logits/probs=...).sample().
    (
        r'(?P<block>'
        r'(?P<indent>[ \t]*).*?Categorical\([^\n]+\)\.sample\(\)[^\n]*\n'
        r'(?P=indent)znn\s*=\s*znn_candidate\[[^\n]+\]\n'
        r')',
        'categorical'
    ),

    # Pattern C: variable named z_nn instead of znn.
    (
        r'(?P<block>'
        r'(?P<indent>[ \t]*)prob\s*=\s*torch\.softmax\([^\n]+\)\n'
        r'(?P=indent).*?torch\.multinomial\([^\n]+\)\.squeeze[^\n]*\n'
        r'(?P=indent)z_nn\s*=\s*znn_candidate\[[^\n]+\]\n'
        r')',
        'multinomial_z_nn'
    ),
]

patched = False

for pat, name in patterns:
    m = re.search(pat, s, flags=re.S)
    if not m:
        continue

    block = m.group("block")
    indent = m.group("indent")
    target_var = "z_nn" if "z_nn" in block and "znn =" not in block else "znn"

    new_block = f'''{indent}if args.top_agg == "mean":
{indent}    {target_var} = znn_candidate.mean(dim=0)
{indent}elif args.top_agg == "nearest":
{indent}    {target_var} = znn_candidate[0]
{indent}else:
'''
    # Indent original block under else.
    new_block += "\n".join(indent + "    " + line[len(indent):] if line.startswith(indent) else indent + "    " + line for line in block.rstrip("\n").split("\n")) + "\n"

    s = s[:m.start("block")] + new_block + s[m.end("block"):]
    print(f"[OK] patched sampling block with pattern: {name}")
    patched = True
    break

if not patched:
    print("[ERROR] Could not auto-patch top-k sampling block.")
    print("Please run:")
    print("  grep -n \"znn_candidate\\|multinomial\\|Categorical\\|topk\\|topk\" -C 5 evalvae_fix_recon.py")
    raise SystemExit(1)

p.write_text(s)
print("[DONE] wrote evalvae_fix_recon_topkmean.py")
