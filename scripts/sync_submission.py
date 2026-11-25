"""Bundle the authoritative morphology implementation/configs into submission/main.py."""
import ast
import json
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parents[1]


def sync():
    path = ROOT / 'submission/main.py'
    text = path.read_text(encoding='utf-8')
    core = (ROOT / 'src/segmentation/postprocess.py').read_text(encoding='utf-8')
    core = core[core.index('def _fill_holes_limited'):].rstrip()
    start = text.index('def _fill_holes_limited')
    end = text.index('def segment_image', start)
    text = text[:start] + core + '\n\n\n' + text[end:]
    configs = [json.loads((ROOT / 'config' / name).read_text(encoding='utf-8'))
               for name in ('pipeline.json', 'baseline.json')]
    replacements = dict(POSTPROCESS=pformat(configs[0]['postprocess'], sort_dicts=False),
                        BASELINE_POSTPROCESS=pformat(configs[1]['postprocess'], sort_dicts=False))
    lines = text.splitlines(keepends=True)
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in replacements]
    present = set()
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        name = node.targets[0].id
        lines[node.lineno-1:node.end_lineno] = [name + ' = ' + replacements[name] + '\n']
        present.add(name)
    text = ''.join(lines)
    if 'BASELINE_POSTPROCESS' not in present:
        index = text.index('DEMO_IDS =')
        text = text[:index] + 'BASELINE_POSTPROCESS = ' + replacements['BASELINE_POSTPROCESS'] + '\n' + text[index:]
    text = text.replace('postprocess(skin, POSTPROCESS)',
                        'postprocess(skin, BASELINE_POSTPROCESS if diagonal else POSTPROCESS)')
    compile(text, str(path), 'exec')
    path.write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    sync()
