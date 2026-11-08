"""Verify that every original tracked file still matches the saved baseline."""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]


def main():
    expected = json.loads((PROJECT / 'member_c/protected_files_sha256.json').read_text())
    mismatches = []
    for relative, digest in expected.items():
        path = PROJECT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(relative)
    if mismatches:
        print('Original files differ from the recorded baseline:')
        print('\n'.join(mismatches))
        return 1
    print(f'All {len(expected)} original tracked files are unchanged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
