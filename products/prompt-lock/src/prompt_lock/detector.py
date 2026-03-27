"""Git-based detection of changed prompt files."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Optional


class ChangedPromptDetector:
    """Detects which prompt files have changed relative to a base git ref."""

    def __init__(self, repo_path: str | Path = "."):
        import git

        self.repo = git.Repo(repo_path, search_parent_directories=True)
        self.repo_root = Path(self.repo.working_dir)

    def changed_files(self, base_ref: str = "HEAD~1") -> list[Path]:
        """Return paths of files changed since base_ref."""
        try:
            base_commit = self.repo.commit(base_ref)
            diff = self.repo.head.commit.diff(base_commit)
            changed = set()
            for d in diff:
                if d.a_path:
                    changed.add(Path(d.a_path))
                if d.b_path and d.b_path != d.a_path:
                    changed.add(Path(d.b_path))
            return list(changed)
        except Exception:
            # First commit, shallow clone, or invalid ref — treat all tracked files as changed
            return [Path(item.path) for item in self.repo.index.entries]

    def filter_by_patterns(self, files: list[Path], patterns: list[str]) -> list[Path]:
        """Return only files that match at least one glob pattern."""
        matched = []
        for f in files:
            for pattern in patterns:
                if fnmatch.fnmatch(str(f), pattern):
                    matched.append(f)
                    break
        return matched

    def detect_changed_prompts(
        self,
        prompt_patterns: list[str],
        base_ref: str = "HEAD~1",
        all_prompts: bool = False,
    ) -> list[Path]:
        """Main entry: return prompt files to evaluate this run.

        If all_prompts=True, returns all files matching the patterns regardless of git changes.
        Otherwise, returns only changed files that match the patterns.
        """
        if all_prompts:
            matched = []
            for pattern in prompt_patterns:
                matched.extend(self.repo_root.glob(pattern))
            return list(set(matched))

        changed = self.changed_files(base_ref)
        return self.filter_by_patterns(changed, prompt_patterns)
