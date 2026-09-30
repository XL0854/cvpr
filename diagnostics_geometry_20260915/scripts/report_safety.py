"""Summarize fixed safety diagnostic and flag teacher/proxy outliers."""
import bootstrap
from bootstrap import DIAG, ROOT
import json, hashlib, html
from collections import Counter
import numpy as np

p=DIAG/'runs/safety100';old=DIAG/'runs/pilot100'
groups=json.loads((p/'summary.json').read_text())
manifest=json.loads((p/'manifest.json').read_text())['samples']
rows=[json.loads((p/s['id']/'result.json').read_text()) for s in manifest]
lines=['# 后续检验：覆盖与形变约束下的 TPS 收益','',
'这是同一批 100 对开发样本上的事后诊断，未训练网络，未检验未知场景泛化或加速。',
'固定残差缩放档位 1/0.75/0.5/0.25/0；选择满足原重叠、并集、两侧各自有效区域均保留至少 99%，两侧各向异性 P95 分别不超过原值 1.1 倍，且未检测到折叠的最大档位。',
'规则不使用匹配误差、PSNR 或 SSIM；所选结果再用留出点与独立 SIFT 代理评价。阈值是在看到 pilot100 后设定，因此不是独立确认性实验。','',
'| 数据（各50对） | 原残差 ΔSSIM | 约束后 ΔSSIM | 原残差 ΔPSNR | 约束后 ΔPSNR | 独立 SIFT 误差：基线→约束后(px) |',
'|---|---:|---:|---:|---:|---:|']
for d,g in groups.items():
 s=g['mSSIM'];q=g['mPSNR'];e=g['independent']
 lines.append(f"| {d} | {s['raw']-s['baseline']:+.6f} | {s['safe']-s['baseline']:+.6f} | {q['raw']-q['baseline']:+.4f} | {q['safe']-q['baseline']:+.4f} | {e['baseline']:.4f} → {e['safe']:.4f} |")
lines+=['','三种结果使用相同的共同有效区，因此本表可直接比较原残差与约束后收益；其掩膜可能与 pilot100 表略不同。仍不能将共同区指标当作全区域质量，99% 约束也不是视觉自然性的保证。','',
'| 数据 | 残差档位计数 | 最小原重叠保留率 | 最小原并集保留率 | 最大各向异性倍率 | 未满足约束对数 |',
'|---|---|---:|---:|---:|---:|']
for d,g in groups.items():
 lines.append(f"| {d} | {g['scales']} | {g['min_retention']['overlap']:.4%} | {g['min_retention']['union']:.4%} | {g['max_anisotropy_ratio']:.4f} | {g['constraint_failures']} |")
lines+=['','## 离群值与代理指标检查','',
'下列数值是每对的误差中位数变化再等权平均，以及 3px 内对应比例变化。它们用于检查收益是否只来自少量大误差点；SIFT 仍不等于真值。','',
'| 数据 | 独立误差中位数平均变化(px) | 独立对应3px内比例变化 | 独立误差均值下降/上升/不变对数 |',
'|---|---:|---:|---:|']
for d in groups:
 rs=[r for r in rows if r['domain']==d and 'independent' in r]
 delta=[r['independent']['safe']['mean_px']-r['independent']['baseline']['mean_px'] for r in rs]
 md=np.mean([r['independent']['safe']['median_px']-r['independent']['baseline']['median_px'] for r in rs])
 hit=np.mean([r['independent']['safe']['within3']-r['independent']['baseline']['within3'] for r in rs])
 lines.append(f'| {d} | {md:+.4f} | {hit:+.2%} | {sum(x<0 for x in delta)}/{sum(x>0 for x in delta)}/{sum(x==0 for x in delta)} |')
lines+=['','## 逐对图像质量退化','',
'约束后 88 对 SSIM 提升、3 对下降、9 对不变；后者包括 5 对此前对应不足的样本及 4 对残差回退为零的样本。几何约束通过不等于质量安全保证。','',
'| 样本 | ΔSSIM | ΔPSNR(dB) | 残差档位 |','|---|---:|---:|---:|']
for r in rows:
 q=r['image_metrics'];ds=q['safe']['mSSIM']-q['baseline']['mSSIM']
 if ds<0:
  lines.append(f"| {r['id']} | {ds:+.6f} | {q['safe']['mPSNR']-q['baseline']['mPSNR']:+.4f} | {r['scale']} |")
lines+=['','## 解释与下一步','',
'- 本次受约束结果仍改善平均独立对应误差与共同区域图像质量，支持继续小规模几何监督研究。此结论仅限当前开发集和诊断口径。',
'- 残差是逐样本拟合得到的，缩放检查还需多次几何计算；这不是拟发表的高效推理方案，也不能据此报告模型加速比。',
'- 本检验不证明 RoMa 监督优于更便宜的 SIFT/传统匹配，也不证明置信度校准有效。后续训练必须加入等预算对照。',
'- 全域无折叠、自然性、原分辨率覆盖、独立场景泛化与教师人工核验仍未完成。','']
robust={}
for domain in groups:
 rs=[r for r in rows if r['domain']==domain and 'independent' in r]
 changes=np.array([r['independent']['safe']['mean_px']-r['independent']['baseline']['mean_px'] for r in rs])
 rng=np.random.default_rng(20260916)
 means=changes[rng.integers(len(changes),size=(5000,len(changes)))].mean(1)
 # Remove the five pairs with the highest BASELINE error, not those with best gains.
 rest=sorted(rs,key=lambda r:r['independent']['baseline']['mean_px'])[:-5]
 robust[domain]=dict(exploratory_pair_bootstrap_ci95=np.quantile(means,[.025,.975]).tolist(),
  exclude_five_largest_baseline_error_mean_change=float(np.mean([r['independent']['safe']['mean_px']-r['independent']['baseline']['mean_px'] for r in rest])),
  note='Post-hoc sensitivity analysis; pairs are not guaranteed scene-independent.')
(p/'robustness.json').write_text(json.dumps(robust,indent=2))
lines+=['## 事后敏感性分析','', '这些是图像对级探索性区间，不能代替场景独立统计。删除项按基线误差选择，未按改善幅度挑选。','',
 '| 数据 | 独立误差变化的配对 bootstrap 95% 区间(px) | 去掉基线误差最大5对后平均变化(px) |','|---|---:|---:|']
for domain,r in robust.items():
 lo,hi=r['exploratory_pair_bootstrap_ci95']
 lines.append(f"| {domain} | [{lo:+.4f}, {hi:+.4f}] | {r['exclude_five_largest_baseline_error_mean_change']:+.4f} |")
lines.append('')
originals=json.loads((DIAG/'runs/preflight/report.json').read_text())['source_sha256']
changed=[]
for path,expected in originals.items():
 h=hashlib.sha256()
 with (ROOT/path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 if h.hexdigest()!=expected:changed.append(path)
check=dict(files_checked=len(originals),changed=changed,unchanged=not changed)
(p/'source_integrity.json').write_text(json.dumps(check,indent=2))
if changed:raise RuntimeError(str(changed))
lines.append(f"核对原源码与权重 {len(originals)} 个，变化 0。")
(p/'REPORT.md').write_text('\n'.join(lines))
cards=[]
for r in rows:
 ident=html.escape(r['id']);prev='../pilot100/pairs/'+ident
 cards.append(f'<details><summary>{ident} · scale={r["scale"]}</summary><p>基线 / 原残差 / 约束后</p>'+''.join(f'<img loading="lazy" src="{src}">' for src in [prev+'/before_fusion.jpg',prev+'/after_fusion.jpg',ident+'/safe_fusion.jpg'])+'</details>')
(p/'REVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>TPS约束复核</title><style>body{font-family:sans-serif;margin:24px}img{width:32%;vertical-align:top}details{padding:12px}summary{cursor:pointer}</style><h1>同一对图像：基线、原残差、约束后</h1>'+''.join(cards))
print('REPORT_READY',p/'REPORT.md')
