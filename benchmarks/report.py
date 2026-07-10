import json
import os
from typing import Dict, Any, Optional


class BenchmarkReport:
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            return json.load(f)

    def compare(self, current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        diff = {}
        sections = ['system', 'performance', 'agents']

        for section in sections:
            current_section = current.get(section, {})
            reference_section = reference.get(section, {})

            if section == 'agents':
                all_agents = set(current_section.keys()) | set(reference_section.keys())
                for agent in sorted(all_agents):
                    cur = current_section.get(agent, {})
                    ref = reference_section.get(agent, {})
                    agent_diff = self._diff_values(cur, ref)
                    if agent_diff:
                        diff[f'agent_{agent}'] = agent_diff
            else:
                section_diff = self._diff_values(current_section, reference_section)
                if section_diff:
                    diff[section] = section_diff

        return diff

    def _diff_values(self, current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        diff = {}
        all_keys = set(current.keys()) | set(reference.keys())
        for key in all_keys:
            cur_val = current.get(key)
            ref_val = reference.get(key)
            if cur_val is None and ref_val is None:
                continue
            if cur_val is None:
                diff[key] = {'current': None, 'reference': ref_val, 'delta': None, 'delta_pct': None}
                continue
            if ref_val is None:
                diff[key] = {'current': cur_val, 'reference': None, 'delta': None, 'delta_pct': None}
                continue
            if isinstance(cur_val, (int, float)) and isinstance(ref_val, (int, float)):
                delta = cur_val - ref_val
                delta_pct = ((delta / ref_val) * 100) if ref_val != 0 else None
                diff[key] = {
                    'current': round(cur_val, 4),
                    'reference': round(ref_val, 4),
                    'delta': round(delta, 4),
                    'delta_pct': round(delta_pct, 4) if delta_pct is not None else None,
                }
            elif cur_val != ref_val:
                diff[key] = {
                    'current': cur_val,
                    'reference': ref_val,
                    'delta': 'changed',
                    'delta_pct': None,
                }
        return diff

    def generate_markdown(self, current: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> str:
        lines = ["# Benchmark Report", ""]

        metadata = current.get('metadata', {})
        lines.append("## System Info")
        lines.append(f"- **Git Commit**: {metadata.get('git_commit', 'unknown')}")
        lines.append(f"- **Git Branch**: {metadata.get('git_branch', 'unknown')}")
        lines.append(f"- **Timestamp**: {metadata.get('timestamp', 'unknown')}")
        lines.append(f"- **Agent Setup**: {metadata.get('args', {}).get('agent_setup', 'unknown')}")
        lines.append(f"- **Games**: {metadata.get('args', {}).get('num_games', '?')}  ")
        lines.append(f"- **Steps/Game**: {metadata.get('args', {}).get('steps_per_game', '?')}")
        lines.append("")

        system = current.get('system', {})
        lines.append("## System Resources")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| RSS Start | {system.get('rss_mb_start', 0):.1f} MB |")
        lines.append(f"| RSS End | {system.get('rss_mb_end', 0):.1f} MB |")
        lines.append(f"| VMS Start | {system.get('vms_mb_start', 0):.1f} MB |")
        lines.append(f"| VMS End | {system.get('vms_mb_end', 0):.1f} MB |")
        lines.append(f"| System Memory (avg) | {system.get('system_memory_percent_avg', 0):.1f}% |")
        lines.append(f"| GPU Allocated (peak) | {system.get('gpu_memory_allocated_mb_peak', 0):.1f} MB |")
        lines.append(f"| GPU Max Allocated (peak) | {system.get('gpu_memory_max_allocated_mb_peak', 0):.1f} MB |")
        lines.append("")

        perf = current.get('performance', {})
        lines.append("## Performance")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Duration | {perf.get('total_duration_s', 0):.2f} s |")
        lines.append(f"| Env Step Time (avg) | {perf.get('env_step_time_ms_avg', 0):.2f} ms |")
        lines.append(f"| Env Step Time (median) | {perf.get('env_step_time_ms_median', 0):.2f} ms |")
        if 'env_step_time_ms_std' in perf:
            lines.append(f"| Env Step Time (std) | {perf['env_step_time_ms_std']:.2f} ms |")
        lines.append(f"| Get Actions Time (avg) | {perf.get('get_actions_time_ms_avg', 0):.2f} ms |")
        lines.append("")

        agents = current.get('agents', {})
        if agents:
            lines.append("## Per-Agent Latency")
            lines.append(f"| Agent | Actions | Time/Action (avg) | Time/Action (median) | Avg Batch Size |")
            lines.append(f"|-------|---------|-------------------|---------------------|----------------|")
            for agent_name, stats in sorted(agents.items()):
                lines.append(
                    f"| {agent_name} | {stats.get('total_actions', 0)} "
                    f"| {stats.get('time_per_action_ms_avg', 0):.2f} ms "
                    f"| {stats.get('time_per_action_ms_median', 0):.2f} ms "
                    f"| {stats.get('average_batch_size', 'N/A')} |"
                )
            lines.append("")

        deep_mcts = current.get('deep_mcts', {})
        if deep_mcts:
            lines.append("## Deep MCTS Statistics")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            for key, value in deep_mcts.items():
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.2f} ms |")
                else:
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        if reference:
            diff = self.compare(current, reference)
            if diff:
                lines.append("## Comparison vs Reference")
                lines.append(f"| Metric | Current | Reference | Delta | Delta % |")
                lines.append(f"|--------|---------|-----------|-------|---------|")
                for section_key, section_data in diff.items():
                    for metric, vals in section_data.items():
                        cur = vals.get('current', 'N/A')
                        ref = vals.get('reference', 'N/A')
                        delta = vals.get('delta', 'N/A')
                        pct = vals.get('delta_pct', 'N/A')
                        if isinstance(cur, float):
                            cur = f"{cur:.2f}"
                        if isinstance(ref, float):
                            ref = f"{ref:.2f}"
                        if isinstance(delta, float):
                            delta = f"{delta:+.2f}"
                        if isinstance(pct, float):
                            pct = f"{pct:+.1f}%"
                        lines.append(f"| {section_key}.{metric} | {cur} | {ref} | {delta} | {pct} |")
            else:
                lines.append("## Comparison vs Reference")
                lines.append("No significant differences detected.")
            lines.append("")

        return '\n'.join(lines)
