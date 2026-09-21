from nodes.write_notes import validate_summary_cards
from state import SummaryCards


SOURCE = """
Alpha contribution improves transformer sequence learning with source evidence.
Beta problem limits recurrent parallel training in existing systems.
Gamma method uses self attention encoder decoder architecture.
Delta results improve translation benchmark quality by 0.054 over the baseline.
Epsilon impact enables faster practical parallel modeling and training.
"""


def _cards(**updates):
    values = {
        "tldr": "Alpha contribution improves transformer sequence learning with source evidence.",
        "problem": "Beta problem limits recurrent parallel training in existing systems.",
        "method": "Gamma method uses self attention encoder decoder architecture.",
        "key_results": "Delta results improve translation benchmark quality by 5.4% over the baseline.",
        "why_it_matters": "Epsilon impact enables faster practical parallel modeling and training.",
    }
    values.update(updates)
    return SummaryCards(**values)


def test_validator_keeps_supported_cards_when_one_card_is_missing():
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(key_results=""), source_text=SOURCE
    )

    assert status == "partial"
    assert cards.tldr and cards.problem and cards.method and cards.why_it_matters
    assert cards.key_results == ""
    assert card_statuses["key_results"] == "missing"
    assert card_reasons["key_results"] == "empty_field"
    assert "empty_field:key_results" in reason


def test_validator_blanks_only_the_duplicate_card():
    duplicate = "Beta problem limits recurrent parallel training in existing systems with source evidence."
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(problem=duplicate, method=duplicate), source_text=SOURCE
    )

    assert status == "partial"
    assert cards.problem == duplicate
    assert cards.method == ""
    assert cards.tldr and cards.key_results and cards.why_it_matters
    assert card_statuses["method"] == "invalid"
    assert card_reasons["method"] == "near_duplicate:problem"
    assert "near_duplicate:method" in reason


def test_validator_normalizes_ratio_and_percentage_numbers():
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(_cards(), source_text=SOURCE)

    assert status == "complete"
    assert reason == ""
    assert cards.key_results
    assert card_statuses["key_results"] == "complete"
    assert "key_results" not in card_reasons

    spaced_percent_source = SOURCE.replace("0.054", "5.4 %")
    _, spaced_status, spaced_reason, _, _ = validate_summary_cards(
        _cards(), source_text=spaced_percent_source
    )
    assert spaced_status == "complete"
    assert spaced_reason == ""


def test_validator_blanks_only_card_with_unsupported_number():
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(key_results="Delta results improve translation benchmark quality by 99% over the baseline."),
        source_text=SOURCE,
    )

    assert status == "partial"
    assert cards.key_results == ""
    assert cards.tldr and cards.problem and cards.method and cards.why_it_matters
    assert card_statuses["key_results"] == "unsupported"
    assert card_reasons["key_results"] == "number_not_in_source:99%"
    assert "number_not_in_source:key_results" in reason


def test_validator_preserves_numeric_signs_and_thousands_separators():
    signed_source = SOURCE.replace("0.054", "5.4% and a sample count of 1,000")
    cards, status, _, statuses, reasons = validate_summary_cards(
        _cards(
            key_results=(
                "Delta results report -5.4% translation quality and a sample count of 1."
            )
        ),
        source_text=signed_source,
    )

    assert status == "partial"
    assert cards.key_results == ""
    assert statuses["key_results"] == "unsupported"
    assert reasons["key_results"] == "number_not_in_source:-5.4%"

    _, _, _, _, count_reasons = validate_summary_cards(
        _cards(
            key_results=(
                "Delta results report 5.4% translation quality with a sample count of 1."
            )
        ),
        source_text=signed_source,
    )
    assert count_reasons["key_results"] == "number_not_in_source:1"


def test_validator_accepts_ratio_derived_from_reported_percentage():
    percent_source = SOURCE.replace("0.054", "5.4%")
    cards, status, _, statuses, reasons = validate_summary_cards(
        _cards(
            key_results=(
                "Delta results improve translation benchmark quality by 0.054 over the baseline."
            )
        ),
        source_text=percent_source,
    )

    assert status == "complete"
    assert cards.key_results
    assert statuses["key_results"] == "complete"
    assert "key_results" not in reasons


def test_validator_rejects_all_cards_without_usable_source_evidence():
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(), source_text="Paper title only"
    )

    assert status == "invalid"
    assert not any(cards.model_dump().values())
    assert set(card_statuses.values()) == {"invalid"}
    assert set(card_reasons.values()) == {"insufficient_source_text"}
    assert reason == "insufficient_source_text"


def test_key_results_accepts_percentages_derived_from_reported_baselines():
    source = """
    GauS-SLAM reports ATE-RMSE 0.06 cm on Replica, compared with 0.16 cm for
    GS-ICP and 0.36 cm for SplaTAM. These values are reported in the same table.
    """
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(
            key_results=(
                "Trên Replica, ATE-RMSE 0.06 cm tốt hơn 62.5% so với GS-ICP "
                "và khoảng 83% so với SplaTAM."
            )
        ),
        source_text=source,
    )

    assert cards.key_results
    assert card_statuses["key_results"] == "complete"
    assert "key_results" not in card_reasons
    assert "number_not_in_source:key_results" not in reason


def test_why_it_matters_does_not_require_cross_language_lexical_overlap():
    vietnamese_impact = (
        "Điều này giúp tái dựng bản đồ ổn định hơn và làm hệ thống phù hợp "
        "hơn cho robot hoạt động trong cảnh phức tạp."
    )
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(why_it_matters=vietnamese_impact), source_text=SOURCE
    )

    assert cards.why_it_matters == vietnamese_impact
    assert status == "complete"
    assert card_statuses["why_it_matters"] == "complete"
    assert "why_it_matters" not in card_reasons
    assert "no_source_overlap:why_it_matters" not in reason


def test_why_it_matters_still_rejects_only_its_unsupported_number():
    cards, status, reason, card_statuses, card_reasons = validate_summary_cards(
        _cards(why_it_matters="Ứng dụng thực tế nhanh hơn 99% cho robot."),
        source_text=SOURCE,
    )

    assert status == "partial"
    assert cards.why_it_matters == ""
    assert cards.tldr and cards.problem and cards.method and cards.key_results
    assert card_statuses["why_it_matters"] == "unsupported"
    assert card_reasons["why_it_matters"] == "number_not_in_source:99%"
    assert "number_not_in_source:why_it_matters" in reason
