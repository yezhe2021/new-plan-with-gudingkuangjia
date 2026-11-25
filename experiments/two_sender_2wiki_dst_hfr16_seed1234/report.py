"""Report paired split-versus-joint changes and two-source EM rescues."""
import json
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
METHODS = {'native':'Qwen Native','dst_rank64':'DST + Rank64','heterofold_r16':'HeteroFold-R16'}


def report():
    summary = json.loads((ROOT/'summary.json').read_text())
    cohort = json.loads((ROOT/'cohort.json').read_text())
    rows = [json.loads((ROOT/'per_sample'/(r['id']+'.json')).read_text()) for r in cohort]
    assert len(rows) == summary['n'] == 64
    assert all(r['gold'] == c['answer'] for r,c in zip(rows,cohort))
    lines = ['# 2Wiki 双 Sender 零训练评测', '',
             '在固定的 64 道 compositional 开发题上，每题保留全部 10 篇原始文档。A 和 B 各含一跳支持证据和四篇干扰文档。仅 Qwen 接收问题。单 Sender 联合读取和双 Sender 独立读取使用相同的文档顺序和问题后缀。每项为 EM / F1（%），自由生成上限为 64 个 token。', '',
             '| 方法 | 仅 A | 仅 B | 单路联合读取 | 双路独立读取 | 拆分后的 F1 变化（百分点，配对 95% 区间） |',
             '|---|---:|---:|---:|---:|---:|']
    def interval(delta):
        rng=random.Random(1234)
        draws=sorted(100*mean(rng.choices(delta,k=len(delta))) for _ in range(5000))
        return f'{100*mean(delta):+.2f} [{draws[124]:+.2f}, {draws[4874]:+.2f}]'
    for key,name in METHODS.items():
        s=summary['methods'][key]
        vals=[f"{s[arm]['em']*100:.2f} / {s[arm]['f1']*100:.2f}" for arm in ('a_only','b_only','joint','split')]
        delta=[r['scores'][key]['split']['f1']-r['scores'][key]['joint']['f1'] for r in rows]
        lines.append('| '+' | '.join([name,*vals,interval(delta)])+' |')
    difference=[r['scores']['dst_rank64']['split']['f1']-r['scores']['heterofold_r16']['split']['f1'] for r in rows]
    lines.extend(['','双路独立读取时，DST 减去 HeteroFold 的配对 F1 差为 '+interval(difference)+' 个百分点。','',
                  '| 方法 | 单路 A、B 均错且双路答对 | 联合答对、拆分答错 | 联合答错、拆分答对 |',
                  '|---|---:|---:|---:|'])
    for key,name in METHODS.items():
        s=summary['methods'][key]
        lines.append('| '+' | '.join([name,*[str(s[k]) for k in ('both_sides_wrong_split_correct','joint_correct_split_wrong','joint_wrong_split_correct')]])+' |')
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    report()
