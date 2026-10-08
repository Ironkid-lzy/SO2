"""Deliver immutable EXP-018 evidence and the exploratory five-seed D comparison."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path('/home/lzy/Projects/SO2_t007')
RESEARCH = Path('/home/lzy/Projects/rl_sample_efficiency_research_t007')
WORK = ROOT / '_so2_work/runs/EXP-018'
STEPS = list(range(0, 100001, 2500))
EXPECTED = (1, 2, 3, 4)


def savej(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluations(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row.get('event') == 'evaluation']
    assert [row['env_steps'] for row in rows] == STEPS, path
    scores = []
    for row in rows:
        returns = row['returns']
        assert row['episodes_completed'] == 20 and len(returns) == 20, path
        assert all(math.isfinite(float(x)) for x in returns), path
        mean = statistics.mean(float(x) for x in returns)
        assert math.isclose(mean, float(row['return_mean']), abs_tol=1e-4), path
        scores.append(float(row['normalized_score_100']))
    return scores


def auc(scores):
    return float(np.trapezoid(scores, STEPS) / 100000) if hasattr(np, 'trapezoid') else float(np.trapz(scores, STEPS) / 100000)


def stats(values):
    return {'mean': statistics.mean(values), 'std_sample': statistics.stdev(values) if len(values) > 1 else None}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=RESEARCH, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    accepted = tuple(sorted(state.get('seeds_completed', [])))
    assert set(accepted).issubset(EXPECTED)
    complete = not args.partial and accepted == EXPECTED and state['status'] == 'completed'
    rawbase = RESEARCH / 'results/raw/EXP-018'
    if rawbase.exists():
        raise RuntimeError('EXP-018 research raw path exists; inspect before recovery')
    rawbase.mkdir(parents=True)
    evidence = RESEARCH / 'reports/so2/SO2-T007'
    evidence.mkdir(parents=True, exist_ok=True)
    for src, name in ((ROOT / '_so2_work/validation/EXP-018/preflight.json', 'preflight.json'),
                      (ROOT / '_so2_work/validation/EXP-018/t007_fixed_batch.json', 't007_fixed_batch.json'),
                      (ROOT / 'resources_linux_t007.json', 'resources_linux_t007.json')):
        if src.exists():
            shutil.copy2(src, evidence / name)

    curves = {}
    for seed in range(5):
        curves[('A', seed)] = evaluations(RESEARCH / f'results/raw/EXP-015/EXP-015-s{seed}/metrics.jsonl')
        curves[('B', seed)] = evaluations(RESEARCH / f'results/raw/EXP-017/EXP-017-s{seed}-attempt1/metrics.jsonl')
    old = RESEARCH / 'results/raw/EXP-016/EXP-016-s0-attempt1'
    assert json.loads((old / 'acceptance.json').read_text())['accepted']
    curves[('D', 0)] = evaluations(old / 'metrics.jsonl')

    for seed in accepted:
        run = WORK / f'EXP-018-s{seed}-attempt1'
        acceptance = json.loads((run / 'acceptance.json').read_text())
        meta = json.loads((run / 'run_meta.json').read_text())
        assert acceptance['accepted'] and meta['status'] == 'completed'
        assert meta['git_commit'] == state['code_sha'] and meta['config_sha256'] == state['config_sha256']
        assert meta['checkpoint_sha256'] == state['checkpoint_sha256']
        curves[('D', seed)] = evaluations(run / 'metrics.jsonl')
        raw = rawbase / run.name
        raw.mkdir()
        for name in ('metrics.jsonl', 'run_meta.json', 'policy_reload.json', 'acceptance.json',
                     'live_policy_probes.jsonl', 'formatted_total_config.py', 'total_config.py'):
            if (run / name).exists():
                shutil.copy2(run / name, raw / name)
        for name in (f'seed{seed}.log', f'seed{seed}_acceptance.log', f'seed{seed}_first_learning_check.json'):
            if (WORK / name).exists():
                shutil.copy2(WORK / name, raw / name)
        large = [WORK / f'seed{seed}.log', *sorted((run / 'ckpt').glob('*.pth.tar'))]
        savej(raw / 'artifact_manifest.json', [
            {'path': str(path.resolve()), 'size_bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in large if path.exists()])
        savej(raw / 'queue_state_snapshot.json', state)

    failed = state.get('current_seed')
    if failed is not None and failed not in accepted:
        run = WORK / f'EXP-018-s{failed}-attempt1'
        if run.exists():
            raw = rawbase / run.name
            raw.mkdir()
            for name in ('metrics.jsonl', 'run_meta.json', 'policy_reload.json', 'acceptance.json',
                         'live_policy_probes.jsonl', 'formatted_total_config.py', 'total_config.py'):
                if (run / name).exists():
                    shutil.copy2(run / name, raw / name)
            for name in (f'seed{failed}.log', f'seed{failed}_acceptance.log', f'seed{failed}_first_learning_check.json'):
                if (WORK / name).exists():
                    shutil.copy2(WORK / name, raw / name)
            large = [WORK / f'seed{failed}.log', *sorted((run / 'ckpt').glob('*.pth.tar'))]
            savej(raw / 'artifact_manifest.json', [
                {'path': str(path.resolve()), 'size_bytes': path.stat().st_size, 'sha256': digest(path)}
                for path in large if path.exists()])
            savej(raw / 'queue_state_snapshot.json', state)
    savej(rawbase / 'queue_state.json', state)

    available = (0, *accepted)
    per = []
    for seed in available:
        row = {'seed': seed, 'D_source': 'EXP-016' if seed == 0 else 'EXP-018'}
        for group in ('A', 'B', 'D'):
            row[f'{group}_endpoint'] = curves[(group, seed)][-1]
            row[f'{group}_auc'] = auc(curves[(group, seed)])
        for ref in ('A', 'B'):
            row[f'D_minus_{ref}_endpoint'] = row['D_endpoint'] - row[f'{ref}_endpoint']
            row[f'D_minus_{ref}_auc'] = row['D_auc'] - row[f'{ref}_auc']
        per.append(row)
    processed = RESEARCH / 'results/processed'
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / 'EXP-018-per-seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per[0]))
        writer.writeheader()
        writer.writerows(per)
    columns = {'env_steps': STEPS}
    for group in ('A', 'B', 'D'):
        matrix = np.asarray([curves[(group, s)] for s in available])
        for idx, seed in enumerate(available):
            columns[f'{group}_seed{seed}'] = matrix[idx].tolist()
        columns[f'{group}_mean'] = matrix.mean(0).tolist()
        columns[f'{group}_std_sample'] = matrix.std(0, ddof=1).tolist() if len(available) > 1 else [None] * 41
    with (processed / 'EXP-018-curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i in range(41):
            writer.writerow([columns[key][i] for key in columns])
    groups = {}
    for subset_name, rows in (('five_seed', per), ('new_four_seed', per[1:])):
        if not rows:
            groups[subset_name] = None
            continue
        groups[subset_name] = {key: stats([r[key] for r in rows]) for key in
                               ('A_endpoint', 'B_endpoint', 'D_endpoint', 'A_auc', 'B_auc', 'D_auc',
                                'D_minus_A_endpoint', 'D_minus_B_endpoint', 'D_minus_A_auc', 'D_minus_B_auc')}
    summary = {'status': 'completed' if complete else 'partial', 'exp_id': 'EXP-018',
               'D_sources': {'seed0': 'EXP-016', 'seeds1to4': 'EXP-018'},
               'accepted_new_seeds': list(accepted), 'missing_new_seeds': sorted(set(EXPECTED) - set(accepted)),
               'five_seed_complete': complete, 'per_seed': per, 'aggregate': groups,
               'code_sha': state['code_sha'], 'config_sha256': state['config_sha256'],
               'preflight_sha256': state['preflight_sha256']}
    savej(processed / 'EXP-018-summary.json', summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = {'A': '#3465a4', 'B': '#c43c35', 'D': '#319358'}
    for group in ('A', 'B', 'D'):
        matrix = np.asarray([curves[(group, s)] for s in available])
        ax1.plot(STEPS, matrix.mean(0), label=group, color=colors[group])
        if len(available) > 1:
            sd = matrix.std(0, ddof=1)
            ax1.fill_between(STEPS, matrix.mean(0) - sd, matrix.mean(0) + sd, color=colors[group], alpha=.14)
    for seed in available:
        ax2.plot(STEPS, np.asarray(curves[('D', seed)]) - np.asarray(curves[('B', seed)]), label=f'seed {seed}')
    ax2.axhline(0, color='black', linewidth=.8)
    ax1.set_ylabel('D4RL normalized score × 100')
    ax2.set_ylabel('Paired D − B')
    ax2.set_xlabel('Total training environment steps')
    ax1.legend()
    ax2.legend(ncol=5)
    fig.tight_layout()
    figures = RESEARCH / 'results/figures'
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / 'EXP-018.png', dpi=180)
    plt.close(fig)

    def cell(x):
        return f"{x['mean']:.6f} ± {x['std_sample']:.6f}" if x and x['std_sample'] is not None else '—'
    table = '\n'.join(f"| {r['seed']} | {r['D_source']} | {r['D_endpoint']:.6f} | {r['D_minus_B_endpoint']:+.6f} | {r['D_minus_A_endpoint']:+.6f} | {r['D_auc']:.6f} | {r['D_minus_B_auc']:+.6f} | {r['D_minus_A_auc']:+.6f} |" for r in per)
    agg = groups['five_seed']
    newer = groups['new_four_seed']
    report = f'''# SO2-T007：DroQ-style seeds 1–4 补齐及 A/B/D 对照

> 状态：{'完成' if complete else '部分完成'}。D 汇总使用 EXP-016 seed0 + EXP-018 seeds1–4。研究解释待指挥端审核。

## 设计和来源

D 为 2Q + LayerNorm + Dropout(.005)，复用 T005 配置和原作者 checkpoint 的 online/target heads [0,1]、完整 actor。新增 seeds1–4 每个独立进程从相同作者 checkpoint 初始化，100000 总训练环境步，含 5000 步预热；UTD10、actor 每10次 critic 更新、固定 alpha=.2。A 是 EXP-015 原版 10Q，B 是 EXP-017 2Q 无 LN/Dropout。LN 与 Dropout 联合变化，不能单独归因于 Dropout。所有指标限于 halfcheetah-medium-replay-v2、单一作者 checkpoint 和固定 head 子集。

扩展 D 到多种子是在看到原 D seed0 和 B 五种子结果之后决定，属于探索性后续；五次并非同时事前规划。20 个评估 episodes 不是20次独立训练。配对 seed 不保证各结构使用同一 RNG 轨迹。

## 冻结身份与预检

- Algorithm branch `codex/so2-t007-droq-seeds`，SHA `{state['code_sha']}`；research branch `codex/so2-t007-delivery`。
- T005 Python config SHA256 `{state['config_sha256']}`；预检 SHA256 `{state['preflight_sha256']}`。
- 作者 checkpoint SHA256 `{state['checkpoint_sha256']}`；D4RL 数据 SHA256 `{state['dataset_sha256']}`。
- 固定 batch 检查记录在 `reports/so2/SO2-T007/`；未增加在线环境步。T006 的可选 RNG 快照在本次训练中没有启用。

## 100k 与 0–100k AUC

| Seed | D 来源 | D 100k | D−B 100k | D−A 100k | D AUC | D−B AUC | D−A AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
{table}

五种子 D 100k：{cell(agg['D_endpoint'])}；A {cell(agg['A_endpoint'])}；B {cell(agg['B_endpoint'])}。配对 D−B：{cell(agg['D_minus_B_endpoint'])}，D−A：{cell(agg['D_minus_A_endpoint'])}。D AUC：{cell(agg['D_auc'])}；配对 D−B AUC：{cell(agg['D_minus_B_auc'])}，D−A AUC：{cell(agg['D_minus_A_auc'])}。仅在全部四个新增 seed 验收后，这些数才是完整五种子估计。

新增四种子单列：D 100k {cell(newer['D_endpoint']) if newer else '—'}，配对 D−B {cell(newer['D_minus_B_endpoint']) if newer else '—'}、D−A {cell(newer['D_minus_A_endpoint']) if newer else '—'}；D AUC {cell(newer['D_auc']) if newer else '—'}，配对 D−B AUC {cell(newer['D_minus_B_auc']) if newer else '—'}、D−A AUC {cell(newer['D_minus_A_auc']) if newer else '—'}。

主指标为 100k 终点；AUC 为 0–100k 梯形积分除以100000。均值和样本标准差按训练 seed 计算。没有预设显著性或非劣效检验；回报差也不等于真实 Q 估计误差。

## 验收和产物

新增 accepted seeds：{list(accepted)}；缺失或失败：{summary['missing_new_seeds']}；队列状态：`{state['status']}`，错误：`{state.get('error', '')}`。每个 accepted seed 的 `acceptance.json` 核验 100000 训练环境步、950000 critic / 95000 actor / 0 alpha 更新、41×20 有限原始回报、checkpoint 重载；退出码和首次学习窗口见队列及各 raw。EXP-016 seed0 的旧 raw 和报告原样保留，仅重新读取作汇总。

完整曲线、逐 seed 表和聚合 JSON 在 `results/processed/EXP-018-*`，A/B/D 曲线图在 `results/figures/EXP-018.png`。新增小型原始证据在 `results/raw/EXP-018/`；大型 checkpoint 和详细日志保留 Linux `/home/lzy/Projects/SO2_t007/_so2_work/runs/EXP-018/`，各 `artifact_manifest.json` 记录绝对路径、字节数与 SHA256。未选择最好 checkpoint，未重跑种子。
'''
    (RESEARCH / 'reports/so2/SO2-T007.md').write_text(report)
    with (RESEARCH / 'docs/EXPERIMENT_TRACKER.md').open('a') as stream:
        stream.write(f"\n\n## EXP-018 — SO2-T007 交付状态：{'completed' if complete else 'partial'}\n\n")
        stream.write('| EXP-ID | Run ID | Seed | Status | D 100k | D AUC |\n|---|---|---:|---|---:|---:|\n')
        for row in per[1:]:
            stream.write(f"| EXP-018 | EXP-018-s{row['seed']}-attempt1 | {row['seed']} | accepted | {row['D_endpoint']:.6f} | {row['D_auc']:.6f} |\n")
        for seed in sorted(set(EXPECTED) - set(accepted)):
            stream.write(f'| EXP-018 | EXP-018-s{seed}-attempt1 | {seed} | missing/partial | — | — |\n')
        stream.write('\nD aggregate: EXP-016 seed0 + accepted EXP-018 seeds1–4. See reports/so2/SO2-T007.md.\n')
    with (RESEARCH / 'docs/RESEARCH_LOG.md').open('a') as stream:
        stream.write(f"\n\n### 2026-10-07 EXP-018 / SO2-T007：{'完成' if complete else '部分交付'}\n")
        stream.write(f"D 补齐 seeds1–4，accepted {list(accepted)}，missing {summary['missing_new_seeds']}。这是观察 seed0/B 后的探索性扩展；五种子结论需完整验收。报告见 reports/so2/SO2-T007.md。\n")
    git('add', 'docs/EXPERIMENT_TRACKER.md', 'docs/RESEARCH_LOG.md', 'reports/so2/SO2-T007.md',
        'reports/so2/SO2-T007', 'results/processed/EXP-018-per-seed.csv',
        'results/processed/EXP-018-curves.csv', 'results/processed/EXP-018-summary.json',
        'results/figures/EXP-018.png')
    git('add', '-f', 'results/raw/EXP-018')
    subprocess.check_call(['git', 'commit', '-m', 'EXP-018: deliver T007 D seed comparison' if complete else 'EXP-018: deliver T007 partial evidence'], cwd=RESEARCH)
    subprocess.check_call(['git', 'push', 'origin', 'codex/so2-t007-delivery'], cwd=RESEARCH)
    print(json.dumps({'status': summary['status'], 'research_sha': git('rev-parse', 'HEAD'), 'accepted_seeds': list(accepted)}))


if __name__ == '__main__':
    main()
