"""CPU-only finalization after the user omitted OOD sample14; retain original protocol."""
import fcntl
import json
from pathlib import Path

import run_ood as run

ROOT = Path(__file__).resolve().parent
request = run.read(ROOT / 'user_stop_request.json')
assert request['effective_expected_samples'] == 37
rows = [run.read(p) for p in (ROOT / 'per_sample').rglob('*.json')]
assert len(rows) == 37 and all(len(r['arms']) == 13 for r in rows)
assert all(r['id'] != request['omitted_sample'] for r in rows)
summary = run.summary(partial=True)
assert summary['groups']['hellaswag']['samples'] == 24
assert summary['groups']['mmlu_pro']['samples'] == 13
summary.update(status='completed', expected_samples=37, original_expected_samples=38,
    completion_scope='user_requested_omit_last_sample', scope_override=request,
    original_manifest_unchanged=True, full_original_planned_sample_set=False)
run.save('summary.json', summary)
run.report(summary)
report = ROOT / 'REPORT.md'
notice = ('> 用户于北京时间2026-10-15 14:28要求完成第13条后跳过最后一条。'
          '本报告使用HellaSwag24条、MMLU-Pro13条，共37条；'
          '第14条psychology的2个已完成条件单独保存在excluded_partial，不计分。'
          '原始config/manifest保留；范围变更记录在user_stop_request.json。\n\n')
report.write_text(notice + report.read_text(encoding='utf-8'), encoding='utf-8')

# Regenerate only this experiment's shared section with the effective sample count.
p = Path(run.CFG['shared_results'])
begin = '<!-- BEGIN ' + run.CFG['experiment_name'] + ' -->'
end = '<!-- END ' + run.CFG['experiment_name'] + ' -->'
lines = [begin, '## ' + run.CFG['experiment_name'], '',
    '- Server B; status: completed within user-requested scope.', '- Timestamp: ' + run.now(),
    '- Effective scope: HellaSwag24 + MMLU-Pro13 =37 completed samples,13 conditions each.',
    '- User omitted final psychology sample; its2 partial conditions are excluded and preserved.',
    '- Original config/manifest unchanged; scope override: `' + str(ROOT / 'user_stop_request.json') + '`',
    '- Summary: `' + str(ROOT / 'summary.json') + '`; report: `' + str(report) + '`',
    '- Frozen ID PCA; original official requests, filters and budgets; no OOD refit.']
for group, entry in summary['groups'].items():
    lines.append('- ' + group + ': ' + json.dumps({arm: {'score': value['score'],
        'drop_from_native': value['paired_drop_from_native']} for arm, value in entry['arms'].items()}))
section = '\n'.join(lines + ['', end, ''])
with (p.parent / '.experiment_results.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    text = p.read_text(encoding='utf-8')
    if begin in text:
        start = text.index(begin); stop = text.index(end, start) + len(end)
        text = text[:start] + section + text[stop:]
    else:
        text += '\n' + section
    tmp = p.with_suffix('.md.tmp'); tmp.write_text(text, encoding='utf-8'); tmp.replace(p)
run.save('shared_summary_update.json', {'status': 'completed', 'effective_samples': 37,
    'completion_scope': summary['completion_scope'], 'timestamp': run.now()})
run.save('finalization.json', {'status': 'completed', 'timestamp': run.now(),
    'script_sha256': run.sha(ROOT / 'finalize_user_stop.py'),
    'original_manifest_sha256': run.sha(ROOT / 'manifest.json'),
    'summary_sha256': run.sha(ROOT / 'summary.json'), 'request': request})
run.save('status.json', {'status': 'completed', 'completion_scope': summary['completion_scope'],
    'completed_samples': 37, 'expected_samples': 37, 'original_expected_samples': 38,
    'omitted_sample': request['omitted_sample'], 'timestamp': run.now()})
print('Finalized37 samples: HellaSwag24 + MMLU-Pro13; all13 conditions complete; final sample excluded.')
