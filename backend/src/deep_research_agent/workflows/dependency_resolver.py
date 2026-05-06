from __future__ import annotations

from collections import defaultdict, deque

from .contracts import WorkflowDependencyGraph, WorkflowStageDefinition


def build_dependency_graph(stages: list[WorkflowStageDefinition]) -> WorkflowDependencyGraph:
    nodes = [stage.stage_id for stage in stages]
    node_set = set(nodes)
    edges: list[tuple[str, str]] = []
    missing: list[dict[str, str]] = []
    for stage in stages:
        for dep in stage.depends_on:
            if dep in node_set:
                edges.append((dep, stage.stage_id))
            else:
                missing.append({"stage_id": stage.stage_id, "missing_dependency": dep})
        for dep in stage.optional_depends_on:
            if dep in node_set:
                edges.append((dep, stage.stage_id))
    graph = WorkflowDependencyGraph(nodes=nodes, edges=edges, missing_dependencies=missing)
    graph.cycles_detected = detect_cycles(graph)
    graph.execution_layers = compute_execution_layers(stages)
    return graph


def detect_missing_dependencies(stages: list[WorkflowStageDefinition]) -> list[dict[str, str]]:
    return build_dependency_graph(stages).missing_dependencies


def detect_cycles(graph: WorkflowDependencyGraph) -> list[list[str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for source, target in graph.edges:
        adjacency[source].append(target)
    visited: set[str] = set()
    in_stack: set[str] = set()
    path: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        visited.add(node)
        in_stack.add(node)
        path.append(node)
        for child in adjacency.get(node, []):
            if child not in visited:
                visit(child)
            elif child in in_stack and child in path:
                cycles.append([*path[path.index(child) :], child])
        path.pop()
        in_stack.remove(node)

    for node in graph.nodes:
        if node not in visited:
            visit(node)
    return cycles


def topological_sort(stages: list[WorkflowStageDefinition]) -> list[str]:
    graph = build_dependency_graph(stages)
    if graph.cycles_detected:
        return []
    incoming = {node: 0 for node in graph.nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in graph.edges:
        incoming[target] += 1
        outgoing[source].append(target)
    queue = deque([node for node in graph.nodes if incoming[node] == 0])
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in outgoing.get(node, []):
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    return order if len(order) == len(graph.nodes) else []


def compute_execution_layers(stages: list[WorkflowStageDefinition]) -> list[list[str]]:
    graph = build_dependency_graph_without_layers(stages)
    if graph.cycles_detected:
        return []
    incoming = {node: 0 for node in graph.nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in graph.edges:
        incoming[target] += 1
        outgoing[source].append(target)
    ready = [node for node in graph.nodes if incoming[node] == 0]
    layers: list[list[str]] = []
    seen: set[str] = set()
    while ready:
        layer = ready
        layers.append(layer)
        next_ready: list[str] = []
        for node in layer:
            seen.add(node)
            for child in outgoing.get(node, []):
                incoming[child] -= 1
                if incoming[child] == 0:
                    next_ready.append(child)
        ready = next_ready
    return layers if len(seen) == len(graph.nodes) else []


def build_dependency_graph_without_layers(
    stages: list[WorkflowStageDefinition],
) -> WorkflowDependencyGraph:
    nodes = [stage.stage_id for stage in stages]
    node_set = set(nodes)
    edges: list[tuple[str, str]] = []
    missing: list[dict[str, str]] = []
    for stage in stages:
        for dep in stage.depends_on:
            if dep in node_set:
                edges.append((dep, stage.stage_id))
            else:
                missing.append({"stage_id": stage.stage_id, "missing_dependency": dep})
        for dep in stage.optional_depends_on:
            if dep in node_set:
                edges.append((dep, stage.stage_id))
    graph = WorkflowDependencyGraph(nodes=nodes, edges=edges, missing_dependencies=missing)
    graph.cycles_detected = detect_cycles(graph)
    return graph


def find_downstream_stages(stage_id: str, stages: list[WorkflowStageDefinition]) -> list[str]:
    graph = build_dependency_graph(stages)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in graph.edges:
        outgoing[source].append(target)
    return _walk(stage_id, outgoing)


def find_upstream_stages(stage_id: str, stages: list[WorkflowStageDefinition]) -> list[str]:
    graph = build_dependency_graph(stages)
    incoming: dict[str, list[str]] = defaultdict(list)
    for source, target in graph.edges:
        incoming[target].append(source)
    return _walk(stage_id, incoming)


def explain_dependency_path(
    from_stage: str,
    to_stage: str,
    stages: list[WorkflowStageDefinition],
) -> list[str]:
    graph = build_dependency_graph(stages)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in graph.edges:
        outgoing[source].append(target)
    queue = deque([(from_stage, [from_stage])])
    seen = {from_stage}
    while queue:
        node, path = queue.popleft()
        if node == to_stage:
            return path
        for child in outgoing.get(node, []):
            if child not in seen:
                seen.add(child)
                queue.append((child, [*path, child]))
    return []


def _walk(start: str, adjacency: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    queue = deque(adjacency.get(start, []))
    seen: set[str] = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        queue.extend(adjacency.get(node, []))
    return out
