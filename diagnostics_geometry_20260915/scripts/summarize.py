"""Reproducible pair-weighted summaries; reports every selected pair and omission."""
import bootstrap
from bootstrap import DIAG, ROOT
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from collections import Counter
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0)
    if not np.isfinite(a).all():raise ValueError('Nonfinite summary values')
    rng=np.random.default_rng(20260915)
    means=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                min=float(a.min()),max=float(a.max()),
                exploratory_pair_bootstrap_ci95=np.quantile(means,[.025,.975]).tolist(),
                negative=int((a<0).sum()),positive=int((a>0).sum()),zero=int((a==0).sum()))


def group(rows):
    out=dict(selected=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
             optimized=sum(r.get('optimization',{}).get('status')=='completed' for r in rows))
    for kind in ['held','independent']:
        available=[r for r in rows if kind in r]
        out[kind]={key:stats([r[kind][key]['mean_px'] for r in available]) for key in ['before','after']}
        out[kind]['change']=stats([r[kind]['paired_mean_change_px'] for r in available])
        out[kind]['median_error_change']=stats([r[kind]['after']['median_px']-r[kind]['before']['median_px'] for r in available])
        out[kind]['missing']=[r['id'] for r in rows if kind not in r]
        out[kind]['inversion_fail_pairs']=sum(r[kind]['after']['inversion_failure']>0 for r in available)
    available=[r for r in rows if r.get('image_metrics',{}).get('before',{}).get('valid') and r['image_metrics']['after']['valid']]
    for metric in ['mSSIM','mPSNR']:
        out[metric]={key:stats([r['image_metrics'][key][metric] for r in available]) for key in ['before','after']}
        out[metric]['change']=stats([r['image_metrics']['after'][metric]-r['image_metrics']['before'][metric] for r in available])
    t=[r for r in rows if 'sift_all' in r.get('teacher',{}) and 'independent' in r]
    out['teacher_sift_error']=stats([r['teacher']['sift_all']['mean_px'] for r in t])
    out['teacher_minus_baseline_on_sift']=stats([r['teacher']['sift_all']['mean_px']-r['independent']['before']['mean_px'] for r in t])
    out['teacher_sift_median_error']=stats([r['teacher']['sift_all']['median_px'] for r in t])
    out['teacher_filtered_fraction']=stats([r['teacher']['filtered_fraction'] for r in rows if 'teacher' in r])
    out['overlap_retention']=stats([r['image_metrics']['overlap_retention'] for r in rows if 'image_metrics' in r])
    out['fold_pairs_after']=sum(any(v['sampled_tps_fold_fraction']>0 or v['triangle_fold_fraction']>0 for v in r.get('final_structure',[])) for r in rows)
    out['fold_pairs_before']=sum(any(v['sampled_tps_fold_fraction']>0 or v['triangle_fold_fraction']>0 for v in r.get('baseline_structure',[])) for r in rows)
    out['anisotropy_ratio']=stats([max(v['local_anisotropy_p95'] for v in r['final_structure'])/
                                  max(v['local_anisotropy_p95'] for v in r['baseline_structure'])
                                  for r in rows if 'final_structure' in r])
    a=[r['support_audit'] for r in rows if r.get('support_audit',{}).get('valid')]
    out['support_audit']={k:stats([r['after'][k]-r['before'][k] for r in a]) for k in ['ssim','psnr']}
    out['support_audit']['retention']=stats([r['retained_baseline_fraction'] for r in a])
    return out


def f(value, digits=4):
    return f'{value:.{digits}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='pilot100');args=p.parse_args()
    run=DIAG/'runs'/args.run
    samples=json.loads((run/'manifest.json').read_text())['samples']
    rows=[]
    for sample in samples:
        folder=run/'pairs'/sample['id'];r=json.loads((folder/'result.json').read_text())
        if (folder/'support_audit.json').exists():r['support_audit']=json.loads((folder/'support_audit.json').read_text())
        rows.append(r)
    previous=json.loads((DIAG/'runs/preflight/report.json').read_text())['source_sha256']
    changed=[k for k,v in previous.items() if not (ROOT/k).exists() or sha(ROOT/k)!=v]
    audit=dict(files_checked=len(previous),changed=changed,unchanged=not changed)
    (run/'source_integrity.json').write_text(json.dumps(audit,indent=2))
    if changed:raise RuntimeError('Original sources/checkpoints changed: '+str(changed))
    groups={domain:group([r for r in rows if r['domain']==domain]) for domain in ['udis_train','classic_development']}
    summary=dict(groups=groups,source_integrity=audit,
        confidence_intervals='Exploratory pair bootstrap, 5000 resamples. Not scene-level intervals: scene IDs unavailable.',
        teacher_evidence='SIFT proxy agreement, not correspondence ground truth; hand verification pending',
        scope='Pair-specific mesh capacity diagnostic; no neural training or generalization established')
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    flat=[]
    for r in rows:
        row=dict(id=r['id'],domain=r['domain'],status=r['status'],train_count=r.get('train_count'),
                 held_count=r.get('held_count'),independent_count=r.get('independent_count'))
        for kind in ['held','independent']:
            for side in ['before','after']:
                row[f'{kind}_{side}_mean_px']=r.get(kind,{}).get(side,{}).get('mean_px')
        for metric in ['mSSIM','mPSNR']:
            for side in ['before','after']:row[f'{metric}_{side}']=r.get('image_metrics',{}).get(side,{}).get(metric)
        row['overlap_retention']=r.get('image_metrics',{}).get('overlap_retention')
        row['teacher_sift_mean_px']=r.get('teacher',{}).get('sift_all',{}).get('mean_px')
        row['optimization_status']=r.get('optimization',{}).get('status')
        flat.append(row)
    with (run/'metrics.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    names={'udis_train':'UDIS 训练集诊断','classic_development':'Classic 跨域开发集'}
    lines=['# 几何蒸馏诊断 A / B：100 对样本报告','',
        '**结论边界：这是外部匹配与逐样本 TPS 表达能力的诊断，不是训练后网络的测试结果。**',
        '原始 Network、CoefNetwork 参数均冻结，α=0.5；全局变换、四角系数保持固定。只对每对图像的两侧 TPS 控制点残差作 150 步优化。',
        '100 对全部有记录，运行错误 0；95 对完成优化，5 对对应不足，保留原结果。',
        '所有结果使用 512×512 输入和固定基线画布；不得与原论文原分辨率指标直接比较。','',
        '## 诊断 A：外部匹配是否提供可用几何信息','',
        '独立算法代理为双向 Lowe ratio<0.7 的 SIFT 匹配，经基础矩阵 MAGSAC 过滤；这些点及其周围 8 像素区域不参与拟合。RoMa 和原拼接在相同 SIFT 点上计算双向源图坐标转移误差。','',
        '| 数据 | 有 SIFT 代理的图像对 | RoMa 平均误差(px) | 固定 α 基线平均误差(px) | RoMa 误差更低的对数 |',
        '|---|---:|---:|---:|---:|']
    for d,g in groups.items():
        lines.append(f"| {names[d]} | {g['teacher_sift_error']['n']} | {f(g['teacher_sift_error']['mean'])} | {f(g['independent']['before']['mean'])} | {g['teacher_minus_baseline_on_sift']['negative']} |")
    lines+=['','这是支持继续研究的代理证据，不能称为 teacher 的真实正确率。SIFT 与 RoMa 都可能在重复纹理、遮挡和运动区域出错；人工核验尚未完成。',
        '筛选使用双向 certainty≥0.5、循环误差≤2 像素、8×8 空间均衡；其阈值在运行前固定，未按结果调参。','',
        '## 诊断 B：现有 TPS 能否利用对应信息','',
        '留出点按源图空间单元划分，约 25% 单元留出；拟合点与留出点、SIFT 代理点在两张图上均相距至少 8 像素。留出点不参与损失、停止条件或最优迭代选择，报告固定预算的最后一步。','',
        '| 数据 | 完成优化/总数 | 留出误差前→后(px) | 独立 SIFT 误差前→后(px) | ΔmSSIM | ΔmPSNR(dB) |',
        '|---|---:|---:|---:|---:|---:|']
    for d,g in groups.items():
        lines.append(f"| {names[d]} | {g['optimized']}/{g['selected']} | {f(g['held']['before']['mean'])} → {f(g['held']['after']['mean'])} | {f(g['independent']['before']['mean'])} → {f(g['independent']['after']['mean'])} | {g['mSSIM']['change']['mean']:+.6f} | {g['mPSNR']['change']['mean']:+.4f} |")
    lines+=['','图像指标在优化前后共同有效区域内、腐蚀 3 像素边界后计算。平均按图像对等权，包含保留原结果的未优化样本；无对应点的样本只在对应指标中列为缺失，不当成零误差。','',
        '| 数据 | 有留出点/独立代理的对数 | 留出误差下降对数 | 独立代理误差下降对数 | SSIM 改善/退化/不变 |',
        '|---|---:|---:|---:|---:|']
    for d,g in groups.items():
        s=g['mSSIM']['change'];lines.append(f"| {names[d]} | {g['held']['change']['n']}/{g['independent']['change']['n']} | {g['held']['change']['negative']} | {g['independent']['change']['negative']} | {s['positive']}/{s['negative']}/{s['zero']} |")
    lines+=['','## 退化与覆盖审查','',
        '| 数据 | 采样检测到折叠的对数（前/后） | 重叠面积保留率均值/最小值 | 最坏局部各向异性比例（后/前） |',
        '|---|---:|---:|---:|']
    for d,g in groups.items():
        lines.append(f"| {names[d]} | {g['fold_pairs_before']}/{g['fold_pairs_after']} | {g['overlap_retention']['mean']:.2%}/{g['overlap_retention']['min']:.2%} | {f(g['anisotropy_ratio']['max'])} |")
    lines+=['','未检测到折叠不等于已证明全域无折叠或结构完全自然：检查覆盖网格三角形及每单元 9 个 TPS 导数采样点。面积损失、局部拉伸和视觉失真必须分别考虑。',
        '共同区域可能排除优化后丢失的像素，因此另外做了事后敏感性审查：固定原重叠区域，丢失的像素记 SSIM=0、归一化 MSE=1。此项只测量，不重新优化，也不是标准论文指标。','',
        '| 数据 | 固定原区域、覆盖惩罚后的 ΔSSIM | 覆盖惩罚后的 ΔPSNR(dB) | 已审查对数 |',
        '|---|---:|---:|---:|']
    for d,g in groups.items():
        a=g['support_audit'];lines.append(f"| {names[d]} | {a['ssim'].get('mean',float('nan')):+.6f} | {a['psnr'].get('mean',float('nan')):+.4f} | {a['ssim']['n']} |")
    lines+=['','## 统计不确定性','',
        '下表为按图像对进行 5000 次配对 bootstrap 的探索性 95% 区间。没有场景 ID，不能将其解释为严格场景独立的置信区间。','',
        '| 数据 | Δ独立对应误差(px)，越低越好 | ΔmSSIM，共同区域 |',
        '|---|---:|---:|']
    for d,g in groups.items():
        a=g['independent']['change']['exploratory_pair_bootstrap_ci95'];b=g['mSSIM']['change']['exploratory_pair_bootstrap_ci95']
        lines.append(f'| {names[d]} | [{a[0]:+.4f}, {a[1]:+.4f}] | [{b[0]:+.6f}, {b[1]:+.6f}] |')
    lines+=['','## 局限与后续决策','',
        '- 100 对中 50 对来自 UDIS 训练集；Classic 已在此前研究中被接触，只能视为跨域开发数据。没有使用 UDIS 测试集。',
        '- 去除了可访问旧 manifest 中的图像哈希及 dHash 近重复，但这不是场景级去重，也不能排除其他会话用过 Classic 样本。',
        '- 留出 teacher 匹配仍可能共享 teacher 的系统误差；SIFT 代理不等于人工真值。人工复核入口见 REVIEW.html。',
        '- TPS 最大位移限制为每轴 20 像素（欧氏距离上限约 28.28），不能据未修复的粗对齐失败推断整个 TPS 表达上限。',
        '- 当前证明的是对当前图像对迭代拟合的可能收益，没有证明一个小网络能学会、更没有证明未知场景泛化或训练后加速。',
        '- 固定原画布可能裁掉向外扩展的部分；当前覆盖审查保守统计原区域，不能替代原分辨率完整拼接的自然性检查。',
        '- 对 classic_development_000132 和 classic_development_000129 的有限图像检查显示部分重影改善，但仍有局部弯曲或残余重影；这只是辅助观察，不是人工真值标注。',
        '- 诊断 A 支持外部几何信息可能有用；诊断 B 支持现有 TPS 尚有对齐改进空间。覆盖保持与视觉自然性尚未通过完整验证，因此不能直接判定整体方案成功。',
        '- 覆盖惩罚采用最大归一化像素误差，是事后悲观敏感性检查。其 PSNR 下降不能等同标准 PSNR 退化，但提示收益依赖评价区域，必须报告覆盖代价。',
        '- 下一步应先核验失败/退化与 teacher 离群样本，再做小规模、等训练预算的原损失/增强/几何蒸馏对照；不直接扩大训练。','',
        '## 文件与复现','',
        '- manifest.json / protocol.json：运行前固定的样本和协议。',
        '- metrics.csv / summary.json：逐对与汇总结果。',
        '- pairs/<id>/：teacher 匹配、fit/held 索引、原/优化网格、指标与对比图。',
        '- execution.json：本次运行的脚本哈希与配置。',
        f"- source_integrity.json：核对 {audit['files_checked']} 个原源码/权重文件，改变数量 {len(changed)}。",'']
    (run/'REPORT.md').write_text('\n'.join(lines))
    cards=[]
    for r in rows:
        ident=html.escape(r['id']);folder='pairs/'+ident
        match_preview=(f'<img loading="lazy" src="{folder}/match_review.jpg" alt="匹配抽样">'
                       if (run/'pairs'/r['id']/'match_review.jpg').exists()
                       else '<p>没有可展示的筛选后匹配，保留原拼接结果。</p>')
        delta=r.get('image_metrics',{}).get('after',{}).get('mSSIM',0)-r.get('image_metrics',{}).get('before',{}).get('mSSIM',0)
        cards.append(f'<details><summary>{ident} · {r["status"]} · ΔSSIM {delta:+.4f}</summary>'
                     f'<p>自动匹配尚未经人工核验；不是正确率证明。<a href="{folder}/result.json">完整指标</a></p>'
                     f'{match_preview}'
                     f'<p>优化前 / 后</p><img loading="lazy" class="pair" src="{folder}/before_fusion.jpg">'
                     f'<img loading="lazy" class="pair" src="{folder}/after_fusion.jpg"></details>')
    (run/'REVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>几何诊断复核</title>'
        '<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto}details{padding:12px;border-bottom:1px solid #ddd}'
        'img{max-width:100%}.pair{max-width:49%;vertical-align:top}summary{cursor:pointer}</style>'
        '<h1>100 对几何诊断：匹配及拼接复核</h1><p>这些是逐样本优化结果，不是训练后网络结果。</p>'+''.join(cards))
    if importlib.util.find_spec('matplotlib') is None:
        print('OPTIONAL_PLOT_SKIPPED: matplotlib unavailable; report, HTML and all metrics saved.')
        print('REPORT_READY',run/'REPORT.md')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for d,color in [('udis_train','tab:blue'),('classic_development','tab:orange')]:
        rs=[r for r in rows if r['domain']==d and 'independent' in r]
        x=[r['independent']['paired_mean_change_px'] for r in rs]
        y=[r['image_metrics']['after']['mSSIM']-r['image_metrics']['before']['mSSIM'] for r in rs]
        axes[0].scatter(x,y,label=d,color=color,s=20,alpha=.7)
        rs=[r for r in rows if r['domain']==d and 'sift_all' in r['teacher']]
        axes[1].scatter([r['independent']['before']['mean_px'] for r in rs],
                        [r['teacher']['sift_all']['mean_px'] for r in rs],label=d,color=color,s=20,alpha=.7)
    axes[0].axhline(0,color='gray',lw=.8);axes[0].axvline(0,color='gray',lw=.8)
    axes[0].set(xlabel='Change in independent SIFT transfer error (px)',ylabel='Change in common-support SSIM',title='B: per-pair TPS fitting (not trained inference)')
    axes[1].plot([.1,200],[.1,200],'--',color='gray');axes[1].set(xscale='log',yscale='log',
        xlabel='Baseline transfer error on SIFT (px)',ylabel='RoMa transfer error on SIFT (px)',title='A: independent algorithmic proxy, not ground truth')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(run/'diagnostic_plot.png',dpi=170);plt.close(fig)
    print(json.dumps(groups,indent=2))
    print('REPORT_READY',run/'REPORT.md')


if __name__=='__main__':main()
