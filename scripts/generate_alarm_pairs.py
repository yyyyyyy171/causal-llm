import json
from itertools import combinations
from pathlib import Path


nodes = [
    "HISTORY",
    "CVP",
    "PCWP",
    "HYPOVOLEMIA",
    "LVEDVOLUME",
    "LVFAILURE",
    "STROKEVOLUME",
    "ERRLOWOUTPUT",
    "HRBP",
    "HREKG",
    "ERRCAUTER",
    "HRSAT",
    "INSUFFANESTH",
    "ANAPHYLAXIS",
    "TPR",
    "EXPCO2",
    "KINKEDTUBE",
    "MINVOL",
    "FIO2",
    "PVSAT",
    "SAO2",
    "PAP",
    "PULMEMBOLUS",
    "SHUNT",
    "INTUBATION",
    "PRESS",
    "DISCONNECT",
    "MINVOLSET",
    "VENTMACH",
    "VENTTUBE",
    "VENTLUNG",
    "VENTALV",
    "ARTCO2",
    "CATECHOL",
    "HR",
    "CO",
    "BP"
]


pairs = []

for variable_a, variable_b in combinations(nodes, 2):
    pairs.append({
        "variable_a": variable_a,
        "variable_b": variable_b
    })


print("变量数:", len(nodes))
print("变量对数:", len(pairs))


PROJECT_ROOT = Path(__file__).resolve().parent.parent

output_file = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_pairs.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        pairs,
        f,
        ensure_ascii=False,
        indent=2
    )

print("文件已经保存:")
print(output_file)