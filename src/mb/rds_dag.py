"""
从 bnlearn 的 .rds 文件里读出贝叶斯网络的 DAG。

为什么需要它：方向准确率必须对照真实 DAG。本机确实装了 R
（`D:\R\R-4.6.1`，含 bnlearn），但把 DAG 真值直接读进 Python 管道，
可以让评测这一步不依赖外部程序。好在 `data/<网络>/<网络>.rds` 存的是 `bn.fit` 对象，
每个节点都带 node / parents / children 字段，只要有 parents，整张有向图就确定了。

文件格式：gzip 压缩的 RDS（XDR、序列化版本 2）。踩过两个坑：

1. **版本 2 的 RDS 没有"原生编码字符串"字段**（版本 3 才有）。
   按版本 3 的布局去读会多读一个 int 加若干字节，后面全部错位。
2. **属性存放方式和直觉不同**。对象的属性是一个带 tag 的 pairlist，
   tag 是属性名（names / class），真正的字段名藏在 `names` 属性**的值**里，
   元素的值按那个顺序依次是 node / parents / children / prob。

正确性验证：用 ASIA 校验，解出的 8 条边与已知结构完全一致；
CHILD 20 节点 25 边、INSURANCE 27 节点 52 边、HAILFINDER 56 节点 66 边，
也与文献公布的网络规模吻合。
另外用 bnlearn 独立读同一批 rds，边集与本解析器逐条相同
（见 `scripts/verify_dag_bnlearn.R`）；由 DAG 算出的道德图还与
`<网络>_true_MB_matrix.txt` 逐位一致。
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


# R 序列化类型常量
NILVALUE_SXP = 254
REFSXP = 255
SYMSXP = 1
LISTSXP = 2
CHARSXP = 9
LGLSXP = 10
INTSXP = 13
REALSXP = 14
STRSXP = 16
VECSXP = 19

HAS_ATTR = 1 << 9
HAS_TAG = 1 << 10


class _Reader:

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.refs: list = []

    def int32(self) -> int:
        value = struct.unpack_from(">i", self.data, self.pos)[0]
        self.pos += 4
        return value

    def bytes(self, count: int) -> bytes:
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def double(self) -> float:
        value = struct.unpack_from(">d", self.data, self.pos)[0]
        self.pos += 8
        return value

    def read_item(self):
        flags = self.int32()
        kind = flags & 0xFF

        if kind == NILVALUE_SXP:
            return None

        if kind == REFSXP:
            # 符号等对象会重复出现，第二次起只写引用下标
            return self.refs[(flags >> 8) - 1]

        has_attr = bool(flags & HAS_ATTR)
        has_tag = bool(flags & HAS_TAG)

        if kind == SYMSXP:
            node = ("SYM", self.read_item())
            self.refs.append(node)
            return node

        if kind == CHARSXP:
            length = self.int32()
            if length < 0:              # NA_STRING
                return None
            length &= 0x0FFFFFFF        # 高 4 位是编码标记
            return self.bytes(length).decode("utf-8", errors="replace")

        if kind == LISTSXP:
            tag = self.read_item() if has_tag else None
            head = self.read_item()
            tail = self.read_item()
            if has_attr:
                self.read_item()
            return ("PAIR", tag, head, tail)

        if kind == LGLSXP:
            length = self.int32()
            values = [self.int32() for _ in range(length)]
        elif kind == INTSXP:
            length = self.int32()
            values = [self.int32() for _ in range(length)]
        elif kind == REALSXP:
            length = self.int32()
            values = [self.double() for _ in range(length)]
        elif kind == STRSXP:
            length = self.int32()
            values = [self.read_item() for _ in range(length)]
        elif kind == VECSXP:
            length = self.int32()
            values = [self.read_item() for _ in range(length)]
        else:
            raise ValueError(
                f"未实现的 R 类型: flags={hex(flags)} kind={kind} pos={self.pos}"
            )

        if has_attr:
            return ("VEC_ATTR", values, self.read_item())
        return ("VEC", values)


def _attrs(node) -> tuple[list, list]:
    """把带 tag 的 pairlist 展开成 ([属性名], [属性值])。"""

    names: list = []
    values: list = []

    current = node
    while current and current[0] == "PAIR":
        _, tag, head, tail = current
        names.append(tag[1] if tag else None)
        values.append(head)
        current = tail

    return names, values


def _strings(value) -> list[str]:
    """R 的长度为 1 的字符向量也解析成 ('VEC', ['asia'])，这里拆一层。"""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple) and value and value[0] in ("VEC", "VEC_ATTR"):
        return [item for item in value[1] if isinstance(item, str)]
    return []


def read_rds(path: Path):
    raw = gzip.decompress(Path(path).read_bytes())

    if raw[:2] != b"X\n":
        raise ValueError(f"{path} 不是 XDR 格式的 RDS")

    reader = _Reader(raw)
    reader.pos = 2
    version = reader.int32()
    reader.int32()      # 写入者版本
    reader.int32()      # 最低读取版本

    if version == 3:
        reader.bytes(reader.int32())
    elif version != 2:
        raise ValueError(f"暂不支持的 RDS 版本: {version}")

    return reader.read_item(), version


def extract_dag(path: Path):
    """读出 (节点名列表, {(父, 子)})。"""

    obj, _ = read_rds(path)

    if not isinstance(obj, tuple) or obj[0] not in ("VEC", "VEC_ATTR"):
        raise ValueError("顶层对象不是列表，不像 bn.fit 对象")

    elements = obj[1]

    nodes: list[str] = []
    arcs: set = set()

    for element in elements:

        if not isinstance(element, tuple) or element[0] != "VEC_ATTR":
            continue

        values = element[1]
        tag_names, tag_values = _attrs(element[2])
        attributes = dict(zip(tag_names, tag_values))

        names_node = attributes.get("names")
        if names_node is None:
            continue

        record = dict(zip(names_node[1], values))

        node_name = _strings(record.get("node"))
        if not node_name:
            continue

        name = node_name[0]
        nodes.append(name)

        for parent in _strings(record.get("parents")):
            arcs.add((parent, name))

    return nodes, arcs


def dag_path(network: str):
    """DAG 真值的来源：优先 rds，ALARM 用仓库里的边表。"""

    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2] / "data" / network

    rds = root / f"{network}.rds"
    if rds.exists():
        return rds

    if network == "ALARM":
        graph = _Path(
            r"E:\Causal-Learner-main\Causal-Learner-main\Code\data\alarm_graph.txt"
        )
        if graph.exists():
            return graph

    return None


def load_dag(network: str, variables: list[str]):
    """
    返回 (节点顺序, {(父, 子)})；取不到就返回 (None, None)。

    以数据表头的变量顺序为准重新排布，保证与实验结果的下标一致。
    """

    source = dag_path(network)
    if source is None:
        return None, None

    if source.suffix == ".rds":
        nodes, arcs = extract_dag(source)
    else:
        matrix = np.loadtxt(source, dtype=int)
        nodes = list(variables)
        arcs = {
            (nodes[i], nodes[j])
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if matrix[i, j]
        }

    if set(nodes) != set(variables):
        raise ValueError(
            f"{network}: DAG 的变量名与数据表头不一致，"
            f"多出 {sorted(set(nodes) - set(variables))[:5]}，"
            f"缺少 {sorted(set(variables) - set(nodes))[:5]}"
        )

    ordered = [name for name in variables]
    return ordered, arcs


def moral_adjacency(arcs, variables: list[str]) -> np.ndarray:
    """
    由 DAG 算出道德图邻接矩阵：骨架 + 把每个节点的父节点两两连起来。
    它应当与 `*_true_MB_matrix.txt` 完全一致——这是一个很有用的交叉验证。
    """

    index = {name: i for i, name in enumerate(variables)}
    n = len(variables)
    matrix = np.zeros((n, n), dtype=int)

    parents: dict[str, list[str]] = {}
    for parent, child in arcs:
        matrix[index[parent], index[child]] = 1
        matrix[index[child], index[parent]] = 1
        parents.setdefault(child, []).append(parent)

    for child, parent_list in parents.items():
        for i in range(len(parent_list)):
            for j in range(i + 1, len(parent_list)):
                a = index[parent_list[i]]
                b = index[parent_list[j]]
                matrix[a, b] = 1
                matrix[b, a] = 1

    return matrix
