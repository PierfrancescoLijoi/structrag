"""Build a section tree from blocks and per-block heading levels."""
from __future__ import annotations

from ..ir import Block, Section


def build_tree(title: str, blocks: tuple[Block, ...] | list[Block],
               levels: list[int | None]) -> Section:
    """levels[i] is the heading level of block i (>=1) or None for body text."""
    root = Section(title=title, level=0)
    stack = [root]
    for block, level in zip(blocks, levels):
        if level is None:
            stack[-1].blocks.append(block)
            continue
        while stack[-1].level >= level:
            stack.pop()
        node = Section(title=block.text.strip(), level=level)
        stack[-1].children.append(node)
        stack.append(node)
    return root
