#!/usr/bin/env python3
"""Durable 300-job query-efficiency experiment; no solver math or within-case restoration."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from typing import Any

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}-{time.time_ns()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def append(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        data = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
        position = 0
        while position < len(data):
            position += os.write(descriptor, data[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_dir(path.parent)


def stable(value: Any) -> Any:
    if isinstance(value, float):
        require(math.isfinite(value), 'Non-finite semantic value')
        return {'float_hex': value.hex()}
    if isinstance(value, dict):
        return {str(key): stable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(stable(value), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    provenance = read(root / 'provenance.json')
    manifest = read(root / 'source-manifest.json')
    actual = {path.relative_to(root / 'repo').as_posix(): sha(path)
              for path in (root / 'repo').rglob('*.py')}
    require(actual and actual == manifest, 'Frozen source differs from complete Python manifest')
    source_fp = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    require(source_fp == provenance['source_fingerprint'], 'Unexpected source fingerprint')
    frozen = {'source_revision': provenance['source_revision'], 'source_fingerprint': source_fp,
              'source_python_files': len(actual)}
    for key, name in {'input_sha256': 'input.json', 'adapter_sha256': 'replay_query_efficiency_problem.py',
                      'runner_sha256': 'run_query_efficiency_series.py', 'source_archive_sha256': 'source.tar',
                      'source_manifest_sha256': 'source-manifest.json', 'protocol_sha256': 'protocol.json'}.items():
        frozen[key] = sha(root / name)
        if key in provenance:
            require(provenance[key] == frozen[key], f'Provenance mismatch: {key}')
    frozen['provenance_sha256'] = sha(root / 'provenance.json')
    return frozen


def proc(pid: int) -> dict[str, Any] | None:
    try:
        directory = Path('/proc') / str(pid)
        raw = (directory / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        peak = 0
        for line in (directory / 'status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                peak = int(line.split()[1]) * 1024
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'process_group_id': int(fields[2]), 'session_id': int(fields[3]),
            'start_ticks': int(fields[19]),
            'command': [part.decode(errors='replace') for part in
                        (directory / 'cmdline').read_bytes().split(b'\0') if part],
            'cpu_seconds': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
            'rss_bytes': max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE'),
            'peak_rss_bytes': peak}
    except (OSError, ValueError, IndexError):
        return None


def group_members(group: int) -> list[dict[str, Any]]:
    return [item for directory in Path('/proc').iterdir() if directory.name.isdecimal()
            and (item := proc(int(directory.name))) is not None
            and item['process_group_id'] == group and item['state'] != 'Z']


def main_matches(attempt: dict[str, Any]) -> bool:
    item = proc(attempt['main_pid'])
    return item is not None and item['state'] != 'Z' and (
        item['start_ticks'] == attempt['main_pid_start_ticks']
        and item['process_group_id'] == attempt['process_group_id']
        and item['session_id'] == attempt['process_group_id']
        and item['command'] == attempt['command'])


def owned_group(attempt: dict[str, Any], members: list[dict[str, Any]]) -> bool:
    if not members:
        return True
    if main_matches(attempt):
        return True
    identities = attempt.get('observed_process_identities', {})
    return any(identities.get(str(item['pid'])) == item['start_ticks']
               and item['session_id'] == attempt['process_group_id'] for item in members)


def terminate(attempt: dict[str, Any], process: subprocess.Popen[Any] | None) -> dict[str, Any]:
    group = attempt['process_group_id']
    record: dict[str, Any] = {'process_group_id': group, 'at_utc': now()}
    members = group_members(group)
    require(owned_group(attempt, members),
            'Cannot prove ownership of surviving process group; refuse a new solve. '
            'Inspect the recorded PIDs and resolve their ownership manually before resuming.')
    for sig, seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 3)):
        if not members:
            break
        record[signal.Signals(sig).name + '_at_utc'] = now()
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process is not None:
                process.poll()
            members = group_members(group)
            if not members:
                break
            time.sleep(.1)
    if process is not None:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    record['live_survivor_pids'] = [item['pid'] for item in group_members(group)]
    require(not record['live_survivor_pids'], 'Case process group could not be cleared')
    return record


def journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            require(index == len(lines) - 1, f'Malformed journal line {path}:{index + 1}')
    return result


def progress_snapshot(path: Path) -> dict[str, Any]:
    events = journal(path)
    starts = [event for event in events if event.get('event') == 'planning_start']
    finished = {event.get('planning_id') for event in events
                if event.get('event') in {'planning_end', 'planning_error'}}
    active = next((event for event in reversed(starts)
                   if event.get('planning_id') not in finished), None)
    batches = [event for event in events if event.get('event') == 'parallel_batch_start']
    batch = dict(batches[-1]) if batches else None
    if batch:
        subset = [event for event in events if event.get('batch_id') == batch['batch_id']]
        begun = {event['task_id']: event for event in subset if event['event'] == 'root_task_start'}
        ended = {event['task_id']: event for event in subset if event['event'] == 'root_task_end'}
        count = sum(event.get('root_queries_in_task', 1) for event in ended.values())
        batch.update(root_tasks_started=len(begun), root_tasks_finished=len(ended),
            root_queries_finished=count,
            remaining_root_queries=max(0, batch['root_query_count'] - count),
            active_worker_pids=sorted({event['pid'] for key, event in begun.items() if key not in ended}),
            batch_finished=any(event['event'] in {'parallel_batch_end', 'parallel_batch_error'}
                               for event in subset))
    return {'event_count': len(events), 'active_planning': active, 'last_batch': batch,
            'last_event': events[-1] if events else None,
            'error_count': sum(event.get('event') in {'planning_error', 'parallel_batch_error'}
                              or event.get('event') == 'root_task_end'
                              and event.get('outcome') == 'error' for event in events)}



CONDITIONS = {
    'fixed_depth1': {'depth': 1, 'query_strategy': 'fixed'},
    'fixed_depth2': {'depth': 2, 'query_strategy': 'fixed'},
    'adaptive_rate': {'depth': 2, 'query_strategy': 'reduction_per_query'},
}
CASE_COUNT = 100


def schedule() -> list[dict[str, Any]]:
    names = list(CONDITIONS)
    return [{'job_id': f'{condition}:p{index:03d}', 'condition': condition, 'problem_index': index}
        for index in range(1, CASE_COUNT + 1)
        for condition in names[(index - 1) % 3:] + names[:(index - 1) % 3]]


def generate_input(path: Path, seed: int = 2026101301) -> None:
    """Generate a complete predefined cohort, without importing a solver."""
    require(not path.exists(), 'Refuse to overwrite an existing frozen input')
    import numpy as np
    rng = np.random.default_rng(seed)
    cases = [{'dimension': 5, 'problem_index': index,
              'alternatives': rng.uniform(0.0, 1.0, size=(5, 5)).tolist(),
              'target_weights': rng.dirichlet(np.ones(5, dtype=float)).tolist()}
             for index in range(1, CASE_COUNT + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, {'metadata': {'seed': seed, 'case_count': CASE_COUNT,
        'generator': 'numpy.default_rng PCG64; uniform(0,1) 5x5 then Dirichlet(ones(5))',
        'numpy': np.__version__, 'python': __import__('sys').version,
        'no_case_selection': True, 'all_conditions_share_identical_input': True}, 'cases': cases})


def full_settings(index: int, condition: str) -> dict[str, Any]:
    return {'dimension': 5, 'problem_index': index, 'alternatives': 5,
        **CONDITIONS[condition], 'grid_size': 5, 'root_query_source': 'both',
        'depth_one_query_source': 'ratio', 'workers': 8, 'parallelize_root': True,
        'max_questions': 100, 'volume_confidence_threshold': .99,
        'answer_probability_mode': 'exact_volume', 'progress_per_query': True}


def strict_journal(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    require(data.endswith(b'\n'), f'Completed journal lacks final newline: {path}')
    return [json.loads(line) for line in data.splitlines()]


def validate_result(root: Path, frozen: dict[str, Any], attempt: dict[str, Any],
                    protocol: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    raw = read(root / attempt['result_file'])
    require('error' not in raw and 'result' in raw, f'Adapter failure: {raw.get("error")}')
    condition = attempt['condition']
    require(stable(raw['case']) == stable(case), 'Fixed input case differs')
    metadata = raw['metadata']
    for field in ('source_revision', 'source_fingerprint', 'input_sha256'):
        require(metadata[field] == frozen[field], f'Metadata mismatch: {field}')
    require(metadata['condition'] == condition and metadata['source_label'] == attempt['source_label'],
            'Attempt/condition labels differ')
    require(metadata['multiprocessing_start_method'] == 'forkserver'
            and metadata['thread_environment'] == {key: '1' for key in THREAD_KEYS}, 'Runtime settings differ')
    require(stable(raw['settings']) == stable(full_settings(case['problem_index'], condition)),
            'Full settings differ')
    config = protocol['configurations'][condition]
    require(stable(raw['configuration']) == stable(config)
            and raw['configuration_fingerprint'] == fingerprint(config), 'Full solver configuration differs')
    result = raw['result']
    count, seconds = result['question_count'], result['seconds']
    require(type(count) is int and 0 <= count <= 100 and len(result['queries']) == count
            and len(result['query_seconds']) == count, 'Question counts differ')
    require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0,
            'Invalid solver elapsed time')
    require(result['query_seconds'] == [query['seconds'] for query in result['queries']]
            and all(math.isfinite(value) and value > 0 for value in result['query_seconds'])
            and math.fsum(result['query_seconds']) <= seconds + 1e-6, 'Invalid round timing')
    checkpoints = strict_journal(root / attempt['checkpoint_file'])
    require(checkpoints and checkpoints[0]['event'] == 'solve_start'
            and checkpoints[-1]['event'] == 'solve_end', 'Incomplete solve checkpoints')
    require(checkpoints[0]['condition'] == condition
            and checkpoints[0]['query_strategy'] == CONDITIONS[condition]['query_strategy'],
            'Checkpoint policy differs')
    answers = [event for event in checkpoints if event['event'] == 'answered_query']
    states = [event for event in checkpoints if event['event'] == 'state_boundary']
    require(len(answers) == count and len(states) == count + 1
            and [event['question_index'] for event in answers] == list(range(1, count + 1))
            and [event['answered_query_count'] for event in states] == list(range(count + 1)),
            'Question/state boundary counts differ')
    for event, query in zip(answers, result['queries']):
        require(all(stable(query.get(key)) == stable(value) for key, value in event['query_record'].items()),
                'Actual answer journal differs')
        a, b, value = query['goal_index_a'], query['goal_index_b'], query['value']
        difference = case['target_weights'][a] - value * case['target_weights'][b]
        require(query['answer'] == ('>' if difference > 0 else '<' if difference < 0 else '='),
                'Actual answer differs from fixed target weights')
        if condition == 'adaptive_rate':
            require(query['query_strategy'] == 'reduction_per_query'
                    and query['selected_horizon'] in (1, 2), 'Missing adaptive decision')
            require(all(math.isfinite(query[field]) for field in
                        ('expected_terminal_candidates', 'expected_queries', 'gain', 'rate')),
                    'Non-finite adaptive score')
            require(1 <= query['expected_queries'] <= 2 + 1e-9
                    and math.isclose(query['rate'], query['gain'] / query['expected_queries'],
                                     rel_tol=1e-10, abs_tol=1e-10), 'Adaptive rate arithmetic differs')
            plan_horizons = {plan['horizon'] for plan in query['horizon_best_plans']}
            require(plan_horizons and plan_horizons <= {1, 2}
                    and query['selected_horizon'] in plan_horizons
                    and set(query['horizon_evaluation_seconds']) == {'1', '2'},
                    'Both horizons must be evaluated; retain each nonempty best plan')
    pids = raw['spawned_worker_pids']
    require(len(pids) == len(set(pids)) and all(type(pid) is int for pid in pids) and len(pids) <= 8,
            'Invalid actual pool PID count')
    require(sorted({pid for pool in raw['pool_shutdowns'] for pid in pool['worker_pids']}) == sorted(pids),
            'Worker shutdown receipt differs')
    utilities = [math.fsum(weight * value for weight, value in zip(case['target_weights'], row))
                 for row in case['alternatives']]
    winner = max(range(5), key=lambda index: utilities[index])
    selected = result['selected_candidate']
    require(selected is None or type(selected) is int and 0 <= selected < 5, 'Invalid selection')
    correct = None if selected is None else selected == winner
    require(result['target_winner'] == winner and result['selection_is_correct'] == correct,
            'Winner/correctness differs from independent utility')
    for event in states:
        require(event['is_feasible'], 'Recorded outer state is infeasible')
        candidates = event['candidates']
        volumes, shares = event['candidate_volumes'], event['candidate_volume_shares']
        require(set(volumes) == set(shares) == {str(index) for index in candidates}, 'Volume map keys differ')
        total = math.fsum(volumes.values())
        require(total > 0 and all(math.isfinite(value) and value >= 0 for value in volumes.values()),
                'Invalid stored geometric volumes')
        require(math.isclose(math.fsum(shares.values()), 1, abs_tol=1e-8)
                and all(math.isfinite(value) and 0 <= value <= 1 + 1e-8 for value in shares.values()),
                'Invalid stored shares')
        require(all(math.isclose(shares[key], value / total, rel_tol=1e-7, abs_tol=1e-9)
                    for key, value in volumes.items()), 'Stored volume/share arithmetic differs')
    for stage, event in [('initial', states[0]), ('final', states[-1])]:
        for suffix, field in [('candidates', 'candidates'), ('candidate_volumes', 'candidate_volumes'),
                              ('candidate_volume_shares', 'candidate_volume_shares')]:
            require(stable(result[stage + '_' + suffix]) == stable(event[field]), 'Final/initial checkpoints differ')
    for query, event in zip(result['queries'], states[1:]):
        require(stable(query['remaining_candidates']) == stable(event['candidates'])
                and stable(query['candidate_volumes']) == stable(event['candidate_volumes'])
                and stable(query['candidate_volume_shares']) == stable(event['candidate_volume_shares']),
                'Per-question states differ')
    require(result['solved'] is (selected is not None)
            and result['exactly_solved'] == (len(result['final_candidates']) == 1), 'Stop flags differ')
    require(result['termination_reason'] == ('exact_single_candidate' if result['exactly_solved']
            else 'volume_confidence' if selected is not None else 'max_questions'), 'Stop reason differs')
    require(selected is not None or count == 100, 'Unsolved execution did not reach question guard')
    if selected is not None:
        require(selected in result['final_candidates'] and result['selected_candidate_volume_share'] >= .99,
                'Invalid confidence stop')
    return {key: result[key] for key in ('seconds', 'question_count', 'selected_candidate',
        'target_winner', 'selection_is_correct', 'solved', 'exactly_solved', 'termination_reason',
        'final_candidates', 'selected_candidate_volume_share', 'residual_volume_share')} | {
        'absolute_utility_regret': None if selected is None else utilities[winner] - utilities[selected],
        'spawned_worker_count': len(pids)}


class Series:
    def __init__(self, root: Path, python: Path, timeout: float, retry_failed: bool, lock_fd: int | None = None):
        self.root, self.python, self.timeout, self.retry_failed = root, python, timeout, retry_failed
        self.lock_fd = lock_fd
        self.path = root / 'series-state.json'
        self.state: dict[str, Any] = {}
        self.stop_signal: str | None = None
        self.process: subprocess.Popen[Any] | None = None
        self.active: dict[str, Any] | None = None

    def save(self) -> None:
        self.state['updated_at_utc'] = now()
        atomic(self.path, self.state)

    def event(self, event: str, **fields: Any) -> None:
        append(self.root / 'events.jsonl', {'event': event, 'at_utc': now(), 'pid': os.getpid(), **fields})

    def verified(self) -> None:
        require(freeze(self.root) == self.state['frozen'], 'Frozen source/input/adapter/protocol changed')

    def stop_reason(self) -> str | None:
        if self.stop_signal:
            return self.stop_signal
        path = self.root / 'stop.json'
        if path.exists():
            try:
                return str(read(path).get('reason', 'Manual stop requested'))
            except (OSError, ValueError):
                return 'Unreadable stop.json; stop conservatively'
        return None

    def analyze(self) -> None:
        analyzer = self.root / 'analyze_series.py'
        if not analyzer.exists():
            return
        with (self.root / 'analysis.log').open('ab') as stream:
            try:
                result = subprocess.run([str(self.python), str(analyzer), '--directory', str(self.root)],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
                self.event('analysis_finished', exit_code=result.returncode)
            except (OSError, subprocess.TimeoutExpired) as error:
                self.event('analysis_error', reason=str(error))

    def monitor(self, attempt: dict[str, Any]) -> None:
        members = group_members(attempt['process_group_id'])
        attempt.setdefault('observed_process_identities', {}).update(
            {str(item['pid']): item['start_ticks'] for item in members})
        record = {'at_utc': now(), 'job_id': attempt['job_id'], 'condition': attempt['condition'],
            'problem_index': attempt['problem_index'], 'attempt_number': attempt['attempt_number'],
            'elapsed_process_seconds': time.time() - attempt['started_at_unix'], 'processes': members,
            'process_group_id': attempt['process_group_id'],
            'live_group_cpu_seconds': math.fsum(item['cpu_seconds'] for item in members),
            'live_group_rss_bytes': sum(item['rss_bytes'] for item in members),
            'progress': progress_snapshot(self.root / attempt['progress_file']),
            'answered_question_checkpoint_count': sum(event.get('event') == 'answered_query'
                for event in journal(self.root / attempt['checkpoint_file']))}
        attempt['latest_monitor'] = record
        append(self.root / 'monitoring.jsonl', record)
        self.save()

    def finish(self, entry: dict[str, Any], attempt: dict[str, Any], status: str,
               reason: str | None = None) -> None:
        self.verified()
        attempt.update(cleanup=terminate(attempt, self.process), finished_at_utc=now(),
            process_elapsed_seconds=time.time() - attempt['started_at_unix'],
            exit_code=self.process.returncode if self.process else None)
        if status == 'completed':
            if self.process is None:
                attempt['recovery_successful_report'] = True
            try:
                require(self.process is None or self.process.returncode == 0,
                        f'Process exit code {getattr(self.process, "returncode", None)}')
                summary = validate_result(self.root, self.state['frozen'], attempt, self.protocol,
                                          self.inputs[entry['problem_index']])
                status = 'completed' if summary['solved'] else 'completed_unsolved'
                result_sha = sha(self.root / attempt['result_file'])
                attempt.update(summary=summary, result_sha256=result_sha)
                entry.update(summary=summary, accepted_result_sha256=result_sha,
                             completed_attempt_number=attempt['attempt_number'])
            except (OSError, ValueError, KeyError, TypeError) as error:
                status, reason = 'failed', f'{type(error).__name__}: {error}'
                try:
                    raw = read(self.root / attempt['result_file'])
                    if 'error' in raw:
                        attempt['solver_error'] = raw['error']
                except (OSError, ValueError):
                    pass
        attempt['status'] = entry['status'] = status
        if reason:
            attempt['reason'] = reason
        self.state['active_job_id'] = None
        self.save()
        self.event('attempt_' + status, job_id=entry['job_id'], condition=entry['condition'],
                   problem_index=entry['problem_index'], attempt_number=attempt['attempt_number'],
                   reason=reason, result_sha256=attempt.get('result_sha256'), summary=attempt.get('summary'))
        self.active = self.process = None
        self.analyze()

    def wait(self, entry: dict[str, Any], attempt: dict[str, Any]) -> bool:
        self.active = attempt
        next_monitor = 0.0
        while True:
            alive = self.process.poll() is None if self.process else main_matches(attempt)
            if not alive:
                self.finish(entry, attempt, 'completed')
                return True
            reason = self.stop_reason()
            if reason:
                self.finish(entry, attempt, 'interrupted', reason)
                self.state.update(status='stopped', stop_reason=reason)
                self.save()
                return False
            if self.timeout and time.time() - attempt['started_at_unix'] >= self.timeout:
                self.finish(entry, attempt, 'timed_out', 'Explicit per-job timeout reached')
                return True
            if time.monotonic() >= next_monitor:
                self.monitor(attempt)
                next_monitor = time.monotonic() + 10
            time.sleep(.5)

    def launch(self, entry: dict[str, Any]) -> dict[str, Any]:
        self.verified()
        index, condition = entry['problem_index'], entry['condition']
        parent = self.root / 'attempts' / condition / f'p{index:03d}'
        parent.mkdir(parents=True, exist_ok=True)
        numbers = [int(path.name[1:]) for path in parent.iterdir()
                   if path.is_dir() and path.name.startswith('a') and path.name[1:].isdigit()]
        number = max(numbers, default=0) + 1
        directory = parent / f'a{number:02d}'
        directory.mkdir()
        fsync_dir(parent)
        fsync_dir(parent.parent)
        attempt = {'job_id': entry['job_id'], 'condition': condition, 'problem_index': index,
            'attempt_number': number, 'relative_directory': str(directory.relative_to(self.root)),
            'status': 'launching', 'source_label': f'{condition}-p{index:03d}-a{number:02d}',
            'started_at_utc': now(), 'started_at_unix': time.time()}
        for key, name in (('result_file', 'result.json'), ('progress_file', 'progress.jsonl'),
                          ('checkpoint_file', 'questions.jsonl'), ('log_file', 'run.log')):
            attempt[key] = str((directory / name).relative_to(self.root))
        policy = CONDITIONS[condition]
        command = [str(self.python), str(self.root / 'replay_query_efficiency_problem.py'),
            '--repo-root', str(self.root / 'repo'), '--input-json', str(self.root / 'input.json'),
            '--dimension', '5', '--problem-index', str(index),
            '--output-json', str(self.root / attempt['result_file']),
            '--source-label', attempt['source_label'], '--source-revision', self.state['frozen']['source_revision'],
            '--source-fingerprint', self.state['frozen']['source_fingerprint'],
            '--condition', condition, '--query-strategy', policy['query_strategy'], '--depth', str(policy['depth']),
            '--root-query-source', 'both', '--grid-size', '5',
            '--progress-jsonl', str(self.root / attempt['progress_file']),
            '--checkpoint-jsonl', str(self.root / attempt['checkpoint_file'])]
        attempt['command'] = command
        entry['attempts'].append(attempt)
        entry['status'] = 'running'
        self.state['active_job_id'] = entry['job_id']
        self.save()
        self.active = attempt
        env = os.environ.copy()
        env.pop('OTQA_CPROFILE_DIRECTORY', None)
        env.update({key: '1' for key in THREAD_KEYS})
        with (directory / 'run.log').open('xb') as stream:
            self.process = subprocess.Popen(command, cwd=self.root / 'repo', env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True,
                pass_fds=() if self.lock_fd is None else (self.lock_fd,))
        identity = proc(self.process.pid)
        require(identity is not None, 'Could not record process identity')
        attempt.update(status='running', main_pid=self.process.pid, main_pid_start_ticks=identity['start_ticks'],
            process_group_id=self.process.pid, observed_process_identities={str(self.process.pid): identity['start_ticks']})
        self.save()
        self.event('attempt_start', job_id=entry['job_id'], problem_index=index, condition=condition,
                   attempt_number=number, command=command, main_pid=self.process.pid,
                   case_timeout_seconds=self.timeout)
        return attempt

    def run(self) -> None:
        frozen = freeze(self.root)
        self.protocol = read(self.root / 'protocol.json')
        expected = schedule()
        require(stable(self.protocol['schedule']) == stable(expected)
                and self.protocol['case_timeout_seconds'] == self.timeout, 'Protocol schedule/timeout differs')
        require(self.protocol['conditions'] == CONDITIONS, 'Protocol conditions differ')
        cases = read(self.root / 'input.json')['cases']
        require(len(cases) == 100 and sorted(case['problem_index'] for case in cases) == list(range(1, 101)),
                'Expected all100 fixed cases')
        require(all(case['dimension'] == 5 and len(case['alternatives']) == 5
                and all(len(row) == 5 for row in case['alternatives'])
                and len(case['target_weights']) == 5 for case in cases), 'Input dimensions differ')
        self.inputs = {case['problem_index']: case for case in cases}
        if self.path.exists():
            self.state = read(self.path)
            require(self.state['frozen'] == frozen and self.state['case_timeout_seconds'] == self.timeout
                    and self.state['schedule'] == [job['job_id'] for job in expected], 'Frozen resume mismatch')
        else:
            self.state = {'schema_version': 2, 'status': 'running', 'created_at_utc': now(),
                'case_timeout_seconds': self.timeout, 'frozen': frozen, 'active_job_id': None,
                'schedule': [job['job_id'] for job in expected],
                'jobs': {job['job_id']: dict(job, status='pending', attempts=[]) for job in expected}}
        identity = proc(os.getpid())
        require(identity is not None, 'Linux /proc is required for safe process ownership')
        self.state.update(status='running', controller={'pid': os.getpid(), 'start_ticks': identity['start_ticks'],
            'process_group_id': os.getpgrp(), 'command': identity['command'], 'started_at_utc': now()})
        self.save()
        self.event('controller_start', retry_failed=self.retry_failed, case_timeout_seconds=self.timeout)
        running = [entry for entry in self.state['jobs'].values() if entry['status'] == 'running']
        require(len(running) <= 1, 'Multiple recorded active jobs')
        for entry in self.state['jobs'].values():
            if entry['status'] in {'completed', 'completed_unsolved'}:
                attempt = next(item for item in entry['attempts']
                               if item['attempt_number'] == entry['completed_attempt_number'])
                require(sha(self.root / attempt['result_file']) == entry['accepted_result_sha256']
                        == attempt['result_sha256'], 'Accepted result hash changed')
                summary = validate_result(self.root, frozen, attempt, self.protocol, self.inputs[entry['problem_index']])
                require(stable(summary) == stable(entry['summary']), 'Accepted summary changed')
        if running:
            entry = running[0]
            attempt = entry['attempts'][-1]
            self.active = attempt
            if 'main_pid' not in attempt:
                require((self.root / attempt['result_file']).exists(),
                    'Uncertain launch without PID/report: inheritedlock prevents duplicate while solver runs; '
                    'after lockrelease inspect partialattempt ownership manually, no automatic restart')
                summary = validate_result(self.root, frozen, attempt, self.protocol, self.inputs[entry['problem_index']])
                result_sha = sha(self.root / attempt['result_file'])
                status = 'completed' if summary['solved'] else 'completed_unsolved'
                attempt.update(status=status, summary=summary, result_sha256=result_sha, exit_code=None,
                    recovery_successful_report=True, finished_at_utc=now(),
                    cleanup={'inherited_lock_released': True, 'group_snapshot_unavailable_lost_pid_receipt': True})
                entry.update(status=status, summary=summary, accepted_result_sha256=result_sha,
                             completed_attempt_number=attempt['attempt_number'])
                self.state['active_job_id'] = None
                self.active = None
                self.save()
                self.event('launch_transition_result_recovered', job_id=entry['job_id'])
            elif main_matches(attempt):
                self.event('attempt_adopted', job_id=entry['job_id'])
                if not self.wait(entry, attempt):
                    return
            else:
                require(owned_group(attempt, group_members(attempt['process_group_id'])), 'Unsafe process recovery')
                if (self.root / attempt['result_file']).exists():
                    self.finish(entry, attempt, 'completed')
                else:
                    self.finish(entry, attempt, 'interrupted', 'Restart interrupted problem at preserved case boundary')
        for job_id in self.state['schedule']:
            entry = self.state['jobs'][job_id]
            if entry['status'] in {'completed', 'completed_unsolved'} or (
                    entry['status'] in {'failed', 'timed_out'} and not self.retry_failed):
                continue
            reason = self.stop_reason()
            if reason:
                self.state.update(status='stopped', stop_reason=reason)
                self.save()
                return
            attempt = self.launch(entry)
            if not self.wait(entry, attempt):
                return
        self.verified()
        counts = {status: sum(entry['status'] == status for entry in self.state['jobs'].values())
                  for status in ('completed', 'completed_unsolved', 'failed', 'timed_out', 'interrupted', 'pending', 'running')}
        self.state.update(status='completed' if counts['completed'] + counts['completed_unsolved'] == 300
                          else 'completed_with_failures', finished_at_utc=now(), job_status_counts=counts)
        self.save()
        self.event('controller_finished', status=self.state['status'], job_status_counts=counts)
        self.analyze()

    def fatal(self, error: BaseException) -> None:
        cleanup_error = None
        if self.active and 'process_group_id' in self.active:
            try:
                self.active.update(cleanup=terminate(self.active, self.process), status='interrupted',
                                   finished_at_utc=now(), reason=str(error))
                self.state['jobs'][self.active['job_id']]['status'] = 'interrupted'
            except BaseException as cleanup:
                cleanup_error = repr(cleanup)
        if self.state:
            self.state.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                'traceback': traceback.format_exc(), 'cleanup_error': cleanup_error})
            self.save()
            self.event('controller_error', **self.state['error'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--case-timeout-seconds', type=float, default=0,
                        help='0 means no authorized time limit; positive is an explicit cap')
    parser.add_argument('--retry-failed', action='store_true', help='Explicitly retry errors in preserved newattempts')
    parser.add_argument('--generate-input', type=Path, help='Generate the fixed100-case cohort only; no solves')
    parser.add_argument('--seed', type=int, default=2026101301)
    args = parser.parse_args()
    require(math.isfinite(args.case_timeout_seconds) and args.case_timeout_seconds >= 0, 'Invalid timeout')
    if args.generate_input:
        generate_input(args.generate_input, args.seed)
        return 0
    require(args.root is not None, '--root is required for a benchmark controller')
    root = args.root.resolve()
    python = args.python or root.parent / '20261012-g3-a5-grid10-depth2/.venv/bin/python'
    python = Path(os.path.abspath(python))
    descriptor = os.open(root / 'controller.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another controller owns controller.lock; no job started.', flush=True)
            return 75
        series = Series(root, python, args.case_timeout_seconds, args.retry_failed, descriptor)
        def stop(signum: int, frame: Any) -> None:
            series.stop_signal = f'Controller received {signal.Signals(signum).name}'
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            series.run()
        except BaseException as error:
            series.fatal(error)
            raise
    finally:
        os.close(descriptor)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
