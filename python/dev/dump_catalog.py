import re, pathlib
t = pathlib.Path(r'D:\Desktop\Project\AI4Shader\packages\ir\src\catalog\nodes.ts').read_text(encoding='utf-8')
blocks = re.split(r"(?=kind: ')", t)
want = {'domain_warp','fbm','sdf','grade','hsv','color_ramp','noise','shape','raymarch','palette'}
pat = re.compile(r"\{ name: '([a-zA-Z0-9_]+)', type: '([a-z0-9]+)', label: '([^']*)', default: ([^}]*?)\}")
for b in blocks:
    m = re.match(r"kind: '([a-z0-9_]+)'", b)
    if not m or m.group(1) not in want:
        continue
    print('---', m.group(1), '---')
    for n, tp, lab, d in pat.findall(b):
        print(f'    {n:18s} {tp:7s} {lab:24s} default={d.strip()[:44]}')
# also: node template text for sdf mask and color_ramp stops
for k in ('sdf','color_ramp','grade'):
    for b in blocks:
        m = re.match(r"kind: '([a-z0-9_]+)'", b)
        if m and m.group(1) == k:
            print(f'===== {k} header =====')
            print(b[:900])
