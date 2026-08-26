import os, sys, pathlib
from chai_lab.chai1 import run_inference
tag=sys.argv[1]; OUT="/workspace/docking/output"
IN=f"{OUT}/chai_in_{tag}"; OUTD=f"{OUT}/chai_out_{tag}"; os.makedirs(OUTD, exist_ok=True)
queue=[l.strip() for l in open(f"{OUT}/chai_queue_{tag}.txt") if l.strip()]
done={d for d in os.listdir(OUTD) if os.path.isdir(f"{OUTD}/{d}")}
todo=[r for r in queue if r not in done]
print(f"{tag}: {len(queue)} queued, {len(done)} done, {len(todo)} to do, confidence order", flush=True)
for rec in todo:
    fa=f"{IN}/{rec}.fasta"
    if not os.path.exists(fa): continue
    od=pathlib.Path(f"{OUTD}/{rec}"); od.mkdir(parents=True, exist_ok=True)
    try:
        run_inference(fasta_file=pathlib.Path(fa), output_dir=od,
                      num_trunk_recycles=1, num_diffn_timesteps=80,
                      device='cuda:0', use_esm_embeddings=True)
    except Exception as e:
        open(od/"ERROR.txt","w").write(repr(e))
