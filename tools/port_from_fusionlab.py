"""Copy the XTFT dependency closure from fusionlab into xtft."""

import ast
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])  # fusionlab-learn root
DST = Path(sys.argv[2])  # xtft/src
DRY = "--dry" in sys.argv
SEEDS = [
    "fusionlab.nn.hybrid._xtft",
    "fusionlab.nn.hybrid._super_xtft",
    "fusionlab.nn.hybrid._base_extreme",
    "fusionlab.nn.hybrid.utils",
    "fusionlab.nn.transformers._adj_tft",
    "fusionlab.nn.transformers._tft",
    "fusionlab.nn.components",
    "fusionlab.nn.losses",
    "fusionlab.nn._tensor_validation",
    "fusionlab.nn._utils",
    "fusionlab.nn.anomaly_detection",
    "fusionlab.nn.calibration",
    "fusionlab.nn.callbacks",
    "fusionlab.nn.keras_metrics",
    "fusionlab.nn.custom_metrics",
    "fusionlab.nn._shapes",
    "fusionlab.decorators",
    "fusionlab.exceptions",
    "fusionlab._util",
    "fusionlab._configs",
    "fusionlab._deps",
    "fusionlab._optdeps",
    "fusionlab._dummies",
    "fusionlab.compat.numpy",
    "fusionlab.compat.pandas",
    "fusionlab.compat.sklearn",
    "fusionlab.compat.scipy",
    "fusionlab.compat.tf",
    "fusionlab.compat.keras",
    "fusionlab.compat.keras_fit",
    "fusionlab.compat.types",
    "fusionlab.compat.kt",
    "fusionlab.compat._config",
    "fusionlab.backends.selector",
    "fusionlab.backends.devices",
    "fusionlab.api.docs",
    "fusionlab.api.bunch",
    "fusionlab.api.summary",
    "fusionlab.utils.validator",
    "fusionlab.utils.deps_utils",
    "fusionlab.utils.generic_utils",
    "fusionlab.utils.ts_utils",
    "fusionlab.utils.forecast_utils",
    "fusionlab.utils.data_utils",
    "fusionlab.utils.io_utils",
    "fusionlab.utils.sequence_utils",
    "fusionlab.utils.shapes",
    "fusionlab.utils.calibrate",
    "fusionlab.metrics",
]
# never pulled in: PINN/GeoPrior, tuners, the heavy nn/__init__
EXCLUDE = (
    "fusionlab.nn.pinn",
    "fusionlab.nn.forecast_tuner",
    "fusionlab.nn.utils",
    "fusionlab.utils.nat_utils",
    "fusionlab.utils.subsidence_utils",
    "fusionlab.nn.hybrid._halnet",
)
KEEP_INIT_AS_IS = {
    "fusionlab.compat",
    "fusionlab.metrics",
    "fusionlab.nn.components",
}


def target(mod):
    m = mod
    for a, b in [
        ("fusionlab.nn.", "xtft.nn.tf."),
        ("fusionlab.backends.", "xtft.backends.array."),
        ("fusionlab._fusionlog", "xtft._log"),
    ]:
        if m == a.rstrip("."):
            return b.rstrip(".")
        if m.startswith(a):
            return b + m[len(a) :]
    if m == "fusionlab.nn":
        return "xtft.nn.tf"
    if m == "fusionlab.backends":
        return "xtft.backends.array"
    return "xtft" + m[len("fusionlab") :]


def modpath(mod):
    p = SRC / Path(*mod.split("."))
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def absmod(cur, is_pkg, node):
    if not node.level:
        return node.module or ""
    parts = cur.split(".")
    if not is_pkg:
        parts = parts[:-1]
    parts = parts[: len(parts) - node.level + 1]
    return ".".join(parts + ([node.module] if node.module else []))


def excluded(m):
    return any(m == e or m.startswith(e + ".") for e in EXCLUDE)


seen, stack, missing = {}, list(SEEDS), set()
while stack:
    m = stack.pop()
    if m in seen or excluded(m):
        continue
    p = modpath(m)
    if p is None:
        continue
    seen[m] = p
    is_pkg = p.name == "__init__.py"
    # packages other than KEEP_INIT_AS_IS get fresh, empty inits:
    if is_pkg and m not in KEEP_INIT_AS_IS:
        continue
    tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            t = absmod(m, is_pkg, n)
            if not t.startswith("fusionlab"):
                continue
            if excluded(t):
                missing.add((m, t))
            stack.append(t)
            for a in n.names:
                sub = t + "." + a.name
                if modpath(sub) and not excluded(sub):
                    stack.append(sub)

HEAD = re.compile(r"from\s+(\.+[\w.]*|fusionlab[\w.]*)\s+import")


def rewrite(m, p):
    is_pkg = p.name == "__init__.py"
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            t = absmod(m, is_pkg, n)
            if not t.startswith("fusionlab"):
                continue
            i = n.lineno - 1
            lines[i] = HEAD.sub(
                f"from {target(t)} import", lines[i], 1
            )
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("fusionlab"):
                    i = n.lineno - 1
                    lines[i] = lines[i].replace(a.name, target(a.name))
    return "".join(lines)


tot = 0
for m, p in sorted(seen.items()):
    out = DST / Path(*target(m).split("."))
    out = (
        out / "__init__.py"
        if p.name == "__init__.py"
        else out.with_suffix(".py")
    )
    n = len(
        p.read_text(encoding="utf-8", errors="ignore").splitlines()
    )
    tot += n
    print(f"{n:6d} {m} -> {out.relative_to(DST)}")
    if DRY:
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    if p.name == "__init__.py" and m not in KEEP_INIT_AS_IS:
        if not out.exists():
            out.write_text("", encoding="utf-8")
        continue
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(rewrite(m, p).replace("\r\n", "\n"))
print("TOTAL", len(seen), tot)
for a, b in sorted(missing):
    print("EXCLUDED-REF", a, "->", b)
