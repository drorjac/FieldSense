import re
import sys
from pathlib import Path

def remove_blue(text):
    result = []
    i = 0
    pattern = r'\blue{'
    while i < len(text):
        idx = text.find(pattern, i)
        if idx == -1:
            result.append(text[i:])
            break
        result.append(text[i:idx])
        # collect the content inside \blue{...}
        i = idx + len(pattern)
        depth = 1
        inner_start = i
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        # append inner content (excluding the final closing })
        result.append(text[inner_start:i - 1])
    return ''.join(result)

# Default to the paper/ directory next to this script, so the tool works
# regardless of the working directory it is invoked from.
PAPER_DIR = Path(__file__).resolve().parent.parent / 'paper'
DEFAULT_SRC = PAPER_DIR / 'camera_ready.tex'
DEFAULT_DST = PAPER_DIR / 'camera_ready_clean.tex'


def main(argv):
    src = Path(argv[1]) if len(argv) > 1 else DEFAULT_SRC
    dst = Path(argv[2]) if len(argv) > 2 else DEFAULT_DST

    content = src.read_text(encoding='utf-8')

    # Apply repeatedly until no \blue{} remain (handles nested \blue{})
    while r'\blue{' in content:
        content = remove_blue(content)

    dst.write_text(content, encoding='utf-8')
    print(f'{src} -> {dst}')


if __name__ == '__main__':
    main(sys.argv)



