"""Pure-function coverage for agent/react_agent.py's surviving pieces (the
tool-calling primitives agent/subagent.py reuses -- see tests/test_subagent.py
for the ReAct loop itself, and tests/test_graph.py for the per-hop verify
gate that replaced this module's old standalone finalize/verify/self-correct
driver, which has been removed).
"""

from __future__ import annotations

from deepresearch.agent.react_agent import evaluate_expression


def test_calculate_tool_safe_evaluator():
    """The calculate tool (FRAMES numerical reasoning) evaluates arithmetic /
    comparisons via an AST whitelist and refuses anything else — never eval()."""
    assert evaluate_expression("1985 - 1962") == ("1985 - 1962 = 23", True)
    assert evaluate_expression("max(1972, 1968)") == ("max(1972, 1968) = 1972", True)
    assert evaluate_expression("1972 > 1968") == ("1972 > 1968 = True", True)
    assert evaluate_expression("abs(1962 - 1985)") == ("abs(1962 - 1985) = 23", True)
    # non-arithmetic / unsafe input is refused, not executed
    assert evaluate_expression("__import__('os').system('x')")[1] is False
    assert evaluate_expression("open('f')")[1] is False
