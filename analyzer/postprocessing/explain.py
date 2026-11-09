from __future__ import annotations

from rich.tree import Tree
from rich.text import Text
from .grouping import GroupTrace, GroupBuilder


def renderSelect(tree, sel, verbose=False):
    sel_style = "green" if sel.selected_count > 0 else "red"
    sel_node = tree.add(
        Text(
            f"Select: {sel.pattern} -> "
            f"{sel.selected_count} kept, {sel.dropped_count} dropped",
            style=sel_style,
        )
    )
    if verbose and sel.dropped_items:
        dropped_node = sel_node.add(Text("Dropped:", style="red"))
        for desc in sel.dropped_items:
            dropped_node.add(Text(desc, style="dim red"))
    if verbose and sel.selected_items:
        kept_node = sel_node.add(Text("Kept:", style="green"))
        for desc in sel.selected_items:
            kept_node.add(Text(desc, style="dim green"))


def renderGroup(tree, grp, verbose=False):
    group_node = tree.add(
        Text(
            f"Group by: {grp.pattern} -> {len(grp.groups)} groups",
            style="yellow",
        )
    )
    for cg in grp.groups:
        g_node = group_node.add(
            Text(f"{cg.key} ({cg.count} items)", style="bold yellow")
        )
        if verbose and cg.items:
            for desc in cg.items:
                g_node.add(Text(desc, style="dim"))


def renderSubgroup(tree, subgroup, verbose=False):
    for group_idx, sub in enumerate(subgroup):
        group_label = (
            f"Group {group_idx}" if len(subgroup) > 1 else "Subgroups"
        )
        if isinstance(sub, dict):
            sub_node = tree.add(Text(f"{group_label} (dict):", style="bold blue"))
            for name, sub_trace in sub.items():
                child = renderTrace(sub_trace, label=name)
                sub_node.add(child)
        elif isinstance(sub, list):
            sub_node = tree.add(Text(f"{group_label} (list):", style="bold blue"))
            for i, sub_trace in enumerate(sub):
                child = renderTrace(sub_trace, label=f"[{i}]")
                sub_node.add(child)


def renderTrace(trace: GroupTrace, label, verbose=False) -> Tree:
    tree = Tree(Text(label, style="bold cyan"))

    input_node = tree.add(Text(f"Input: {trace.input.count} items", style="bold"))
    if verbose and trace.input.items:
        for desc in trace.input.items:
            input_node.add(Text(desc, style="dim"))

    if (sel := trace.select) is not None:
        renderSelect(tree, sel, verbose)

    if (grp := trace.grouping) is not None:
        renderGroup(tree, grp, verbose)

    if trace.transforms:
        t_node = tree.add(Text("Transforms:", style="magenta"))
        for step in trace.transforms:
            t_node.add(
                Text(f"{step.name} -> {step.post_count} items after", style="magenta")
            )

    if subgroup := trace.subgroup_traces:
        renderSubgroup(tree, subgroup, verbose)

    return tree
