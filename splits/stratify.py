import numpy as np, glob, csv
OUT="/workspace/datasets/experimental_expansion"
# load bisy rmsd per tool
def load(src):
    m={}
    for f in sorted(glob.glob(f'{OUT}/bisy/bisy_{src}_*.npz')):
        d=np.load(f,allow_pickle=True)
        for s,r in zip(d['system_id'],d['ligand_rmsd']):
            if np.isfinite(r): m[str(s)]=float(r)
    return m
chai=load('chai1'); boltz=load('boltz2'); smina=load('smina')
# metadata: ccd + seq_len from fold index; heavy-atom count from crystal labels lig_offsets
meta={}
for r in csv.DictReader(open(f'{OUT}/exp_fold_index.tsv'),delimiter='\t'):
    meta[r['system_id']]={'ccd':r['ccd'],'seqlen':int(r['seq_len'])}
nheavy={}
for f in sorted(glob.glob(f'{OUT}/crystal_labels/crystal_labels_*.npz')):
    d=np.load(f,allow_pickle=True); lo=d['lig_offsets']; sids=d['system_id']
    for i in range(len(sids)): nheavy[str(sids[i])]=int(lo[i+1]-lo[i])
    del d
COF=set("HEM HEC HEA SAM SAH NAD NAP NDP FAD FMN COA PLP PMP TPP BTN B12 MGD FES SF4".split())
NUC=set("ATP ADP AMP GTP GDP GMP CTP UTP ANP GNP APC ACP XMP IMP".split())
MET=set("FE FE2 ZN MG MN CA CU NI CO FES SF4".split())
def lclass(ccd):
    if ccd in COF: return 'cofactor'
    if ccd in NUC: return 'nucleotide'
    if ccd in MET: return 'metal'
    return 'druglike'
def sizebin(n):
    if n<20: return 'small(<20)'
    if n<35: return 'med(20-34)'
    return 'large(>=35)'
def lenbin(L):
    if L<200: return 'short(<200)'
    if L<400: return 'mid(200-399)'
    return 'long(>=400)'
def report(axis_name, keyfn):
    print(f"\n=== stratify by {axis_name} ===")
    print(f"{'stratum':16s} {'tool':7s} {'n':>6s} {'median':>7s} {'<2A':>5s}")
    strata={}
    for s in set(chai)|set(boltz)|set(smina):
        if s not in meta or s not in nheavy: continue
        k=keyfn(s)
        strata.setdefault(k,{'chai':[],'boltz':[],'smina':[]})
        if s in chai: strata[k]['chai'].append(chai[s])
        if s in boltz: strata[k]['boltz'].append(boltz[s])
        if s in smina: strata[k]['smina'].append(smina[s])
    for k in sorted(strata):
        for t in ['chai','boltz','smina']:
            v=strata[k][t]
            if v: print(f"{k:16s} {t:7s} {len(v):>6d} {np.median(v):>7.2f} {100*np.mean(np.array(v)<2):>4.0f}%")
report("ligand class", lambda s: lclass(meta[s]['ccd']))
report("ligand size (heavy atoms)", lambda s: sizebin(nheavy[s]))
report("protein length", lambda s: lenbin(meta[s]['seqlen']))
