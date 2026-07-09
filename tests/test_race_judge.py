from __future__ import annotations

from eval.race_judge import aggregate_race_score, _match_weight


CRITERIA_DATA = {
    "dimension_weight": {"comprehensiveness": 0.6, "insight": 0.4},
    "criterions": {
        "comprehensiveness": [
            {"criterion": "Breadth of coverage", "explanation": "...", "weight": 0.3},
            {"criterion": "Depth of detail", "explanation": "...", "weight": 0.7},
        ],
        "insight": [
            {"criterion": "Original analysis", "explanation": "...", "weight": 1.0},
        ],
    },
}


def test_aggregate_race_score_weighted_average_within_dimension():
    llm_output = {
        "comprehensiveness": [
            {"criterion": "Breadth of coverage", "analysis": "ok", "target_score": 4.0},
            {"criterion": "Depth of detail", "analysis": "ok", "target_score": 8.0},
        ],
        "insight": [
            {"criterion": "Original analysis", "analysis": "ok", "target_score": 6.0},
        ],
    }
    result = aggregate_race_score(llm_output, CRITERIA_DATA)

    # comprehensiveness: (4*0.3 + 8*0.7) / (0.3+0.7) = (1.2+5.6)/1.0 = 6.8
    assert result["comprehensiveness"] == 6.8
    assert result["insight"] == 6.0
    # overall: 6.8*0.6 + 6.0*0.4 = 4.08 + 2.4 = 6.48
    assert round(result["total"], 4) == 6.48


def test_aggregate_race_score_case_insensitive_and_substring_matching():
    llm_output = {
        "comprehensiveness": [
            {"criterion": "BREADTH OF COVERAGE", "analysis": "ok", "target_score": 10.0},  # case mismatch
            {"criterion": "Depth of detail (partial)", "analysis": "ok", "target_score": 0.0},  # substring
        ],
    }
    result = aggregate_race_score(llm_output, CRITERIA_DATA)
    # (10*0.3 + 0*0.7) / 1.0 = 3.0
    assert result["comprehensiveness"] == 3.0


def test_aggregate_race_score_unmatched_criterion_falls_back_to_average_weight():
    weight = _match_weight("Zebra Xylophone Quokka", {"Breadth of coverage": 0.2, "Depth of detail": 0.8})
    assert weight == 0.5  # average of 0.2 and 0.8, no match found


def test_aggregate_race_score_skips_dimensions_missing_from_criteria():
    llm_output = {
        "comprehensiveness": [{"criterion": "Breadth of coverage", "analysis": "ok", "target_score": 5.0}],
        "readability": [{"criterion": "Clear structure", "analysis": "ok", "target_score": 9.0}],  # not in CRITERIA_DATA
    }
    result = aggregate_race_score(llm_output, CRITERIA_DATA)
    assert "readability" not in result
    assert "comprehensiveness" in result


def test_aggregate_race_score_empty_scores_list_yields_zero_not_crash():
    result = aggregate_race_score({"comprehensiveness": [], "insight": []}, CRITERIA_DATA)
    assert result["comprehensiveness"] == 0.0
    assert result["insight"] == 0.0
    assert result["total"] == 0.0
