"""Display all 27 audited oracle comparisons, including their numerical values."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from comic_plot_style import HEADING

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'data/current_extensions.json'
raw = source.read_bytes()
rows = json.loads(raw)['all_pairs27']
assert len(rows) == 27
benches = [('mmlu_pro', 'MMLU-Pro'), ('math500', 'MATH-500'), ('gsm8k', 'GSM8K')]
pairs = [('qwen', 'llama'), ('qwen', 'mistral'), ('llama', 'mistral')]
ks = [3, 6, 9]
seen = set()
for r in rows:
    key = (r['benchmark'], tuple(r['pair']), r['k'])
    assert key not in seen
    seen.add(key)
    assert r['counts']['coverage'] - r['counts']['member'] == r['counts']['candidate_only']
    assert math.isclose(r['h_mem'], r['counts']['candidate_only'] / r['n'], abs_tol=1e-12)
    assert math.isclose(r['h_mem'], r['coverage'] - r['member_selection'], abs_tol=1e-12)
    assert r['h_mem'] > 0

plt.rcParams.update({'font.family':'Source Sans 3', 'font.size':8,
                    'text.color':'#263440', 'xtick.color':'#263440',
                    'ytick.color':'#263440', 'pdf.fonttype':42,
                    'svg.fonttype':'none', 'svg.hashsalt':'team-pairs-comic-20261025'})
fig = plt.figure(figsize=(3.35, 1.55))
cmap = LinearSegmentedColormap.from_list('missed', ['#FFF8E7','#F2C980','#D79A45'])
matrices = {}
for i, (bench, title) in enumerate(benches):
    ax = fig.add_axes([.115 + .292*i, .36, .265, .44])
    a = np.array([[next(r['h_mem']*100 for r in rows if r['benchmark']==bench and
                       tuple(r['pair'])==pair and r['k']==k) for k in ks] for pair in pairs])
    matrices[bench] = a.tolist()
    im = ax.imshow(a, cmap=cmap, vmin=0, vmax=20, aspect='auto')
    for j in range(3):
        for k in range(3):
            ax.text(k,j,f'{a[j,k]:.1f}',ha='center',va='center',fontsize=8,color='#24313A')
    ax.set_xticks(range(3), ['3','6','9'])
    ax.set_yticks(range(3), ['Q–L','Q–M','L–M'] if i==0 else ['','',''])
    ax.tick_params(length=0,pad=2)
    ax.set_title(title,fontproperties=HEADING,pad=5)
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2)
    ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
fig.text(.55,.19,'Samples per member (k)',ha='center',fontsize=8)
cax=fig.add_axes([.30,.075,.47,.035])
cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,10,20])
cb.ax.tick_params(length=1.5,pad=1,labelsize=7)
cb.outline.set_visible(False)
fig.text(.80,.07,'%',fontsize=8)
provenance='All 27 h_mem point estimates. Percent of evaluation questions. Source SHA256: '+hashlib.sha256(raw).hexdigest()
for ext in ['pdf','svg','png']:
    meta={'Title':'Correct candidates outside all member majorities'}
    meta['Subject' if ext=='pdf' else 'Description']=provenance
    fig.savefig(HERE/f'pair_matrix_comic.{ext}',dpi=300,metadata=meta)
plt.close(fig)
out={'source_sha256':hashlib.sha256(raw).hexdigest(),'rows_checked':27,
     'row_order':pairs,'column_order':ks,'values_percent':matrices,
     'checks':['unique configurations','integer candidate-only counts','oracle differences','all point estimates positive']}
(HERE.parent/'audit/pair_matrix_comic_validation.json').write_text(json.dumps(out,indent=2))
print('PASS: 27 cells match oracle differences and integer counts.')
