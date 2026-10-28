"""Export readable continuation curves and a compact result note."""

import csv
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_extend192_20261019')


def main():
    history = json.loads((ROOT / 'training/history.json').read_text())
    previous = json.loads((Path('/root/autodl-tmp/gaussiangpt_feature_extend_20261019') / 'training/history.json').read_text())
    validations = {h['step']: h['pooled_overlap'] for h in previous + history}
    summary = json.loads((ROOT / 'evaluation/summary.json').read_text())
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    rows = []
    for h in history:
        c = summary['curve'].get(str(h['step']), {})
        rows.append(dict(step=h['step'], validation_overlap=h['pooled_overlap'],
                         training_probe_overlap=h['training_probe_overlap'],
                         raw_ce=h['raw_ce'], trained_feature_rows=h['trained_feature_rows'],
                         online_curve_feature_acceptance=c.get('feature_acceptance'),
                         online_curve_accepted=c.get('feature_accepted'),
                         online_curve_verified=c.get('feature_verified'),
                         online_curve_tokens_per_second=c.get('tokens_per_second')))
    with (out / 'curves.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    biggest = max(zip(history, history[1:]), key=lambda pair: pair[1]['pooled_overlap'] - pair[0]['pooled_overlap'])
    final_rows = []
    for step in [12000, 24000, 48000, 96000, 192000]:
        r = summary['final'][str(step)]
        c = summary['relative_to_12000'].get(str(step))
        final_rows.append(dict(step=step,
                              validation_overlap=validations[step],
                              feature_accepted=r['feature_accepted'], feature_verified=r['feature_verified'],
                              feature_acceptance=r['feature_acceptance'],
                              tokens_per_second=r['tokens_per_second'],
                              speed_source_geomean=c['speed_source_geomean'] if c else 1.,
                              speed_ci95=c['speed_ci95'] if c else None,
                              feature_delta_ci95=c['feature_delta_ci95'] if c else None,
                              relative_to_48000=summary['relative_to_48000'].get(str(step))))
    (out / 'exact_steps.json').write_text(json.dumps(final_rows, indent=2))
    lines = ['# MLP 续训至192000结果摘要', '',
             '本轮从48000继续训练至192000，完整训练与在线评价均完成。以下是精确节点，未使用验证集最优权重替代。', '',
             '| 步数 | 完整验证重合度 | 在线 Feature 接受/验证 | 在线接受率 | token/s | 相对12000来源几何平均 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in final_rows:
        lines.append(f"| {r['step']} | {100*r['validation_overlap']:.4f}% | {r['feature_accepted']}/{r['feature_verified']} | {100*r['feature_acceptance']:.3f}% | {r['tokens_per_second']:.3f} | {r['speed_source_geomean']:.4f}× |")
    a,b = biggest
    lines += ['', f"固定验证曲线的最大相邻增加出现在 {a['step']} → {b['step']}，为 {100*(b['pooled_overlap']-a['pooled_overlap']):.4f} 个百分点。", '',
              '在线中间曲线从48000开始，每4800步一个节点，为三个既有开发来源、一个采样 seed、128-token 窗口；五个精确节点主比较为十个来源、三个 seed、256-token 窗口。两类数字不能直接拼成一条同条件曲线。', '',
              '固定条件重合度、在线 accepted/verified 和实际速度分别解释。训练使用一个 seed，48000模型只有参数权重，因此续训重新初始化一次 AdamW，学习率从1e-6缓降至1e-7，96000处不重启。未进行新来源正式确认、三维导出或画质评价。', '',
              '完整区间见 evaluation/summary.json，逐次结果见 evaluation/records.jsonl。']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    x=[r['step']/1000 for r in rows]
    axes[0].plot(x,[100*r['validation_overlap'] for r in rows],label='Full validation',marker='.',color='#2369a0')
    axes[0].plot(x,[100*r['training_probe_overlap'] for r in rows],label='Fixed training subset',color='#b96b2d')
    axes[0].set(xlabel='Training step (thousands)',ylabel='Distribution overlap (%)',title='Fixed-history evaluation')
    axes[0].legend(fontsize=8)
    curve_rows = [r for r in rows if r['online_curve_feature_acceptance'] is not None]
    axes[1].plot([r['step']/1000 for r in curve_rows],[100*r['online_curve_feature_acceptance'] for r in curve_rows],marker='.',color='#287a56')
    axes[1].set(xlabel='Training step (thousands)',ylabel='Accepted / verified (%)',title='Online curve: 3 sources, 1 seed')
    axes[2].bar(np.arange(5),[100*r['feature_acceptance'] for r in final_rows],color=['#8297a5','#6b9aae','#397e9e','#328378','#256b63'])
    axes[2].set(xticks=np.arange(5),xticklabels=['12k','24k','48k','96k','192k'],xlabel='Exact training step',ylabel='Accepted / verified (%)',title='Main comparison: 10 sources, 3 seeds')
    for i,r in enumerate(final_rows):
        axes[2].text(i,100*r['feature_acceptance']+.3,f"{100*r['feature_acceptance']:.2f}%",ha='center',fontsize=9)
    axes[2].set_ylim(0,max(100*r['feature_acceptance'] for r in final_rows)*1.15)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out/'feature_training_curve.png',dpi=180)
    fig.savefig(out/'feature_training_curve.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
