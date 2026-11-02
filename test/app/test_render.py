from app import render


def code(c, conf, votes=None, **kw):
    votes = votes or {"claude-opus-5-5": [True] * 3, "claude-sonnet-5-5": [True, False, False],
                      "gpt-6.1-terra": [False] * 3}
    return {"code": c, "description": kw.pop("description", f"desc {c}"), "confidence": conf, "votes": votes,
            "n_votes": sum(sum(v) for v in votes.values()), "path": ["Ch9", c], "invalid": kw.pop("invalid", []),
            "excludes1_conflict_with": kw.pop("x1", []), "details": [], **kw}


def result(codes, **kw):
    return {"models": ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-6.1-terra"], "seeds": [0, 1, 2],
            "codes": codes, "chapters": [{"chapter": "Ch9", "description": "Circulatory", "agreement": 1.0}], **kw}


def test_rows_in_order_and_divider_only_with_low_codes():
    html = render.codes_card(result([code("I10", 0.9), code("E78.5", 0.7), code("R73.9", 0.2)]))
    assert html.index("I10") < html.index("E78.5") < html.index("Low agreement") < html.index("R73.9")
    assert "Low agreement" not in render.codes_card(result([code("I10", 0.9)]))


def test_vote_grid_has_nine_cells_with_four_filled():
    html = render.code_row(code("I10", 0.7))
    assert html.count('class="cell') == 9 and html.count('class="cell on"') == 4 and "4/9 runs" in html


def test_reference_pills_and_flags():
    html = render.codes_card(result([code("I10", 0.9, in_reference=True, category_match=True),
                                     code("I11.9", 0.6, in_reference=False, category_match=True),
                                     code("Z56.6", 0.6, in_reference=False, category_match=False,
                                          invalid=["not_found"], x1=["Z56.5"])]))
    assert "✓ In reference" in html and "≈ Same category" in html and "✗ Not in reference" in html
    assert "Invalid code: not_found" in html and "Excludes1 with Z56.5" in html


def test_summary_and_missed_cards():
    r = result([code("I10", 0.9, in_reference=True, category_match=True)],
               missed_reference=[{"code": "E78.5", "description": "Hyperlipidemia"}])
    assert "1/2" in render.summary_card(r) and "E78.5" in render.missed_card(r)
    assert render.missed_card(result([code("I10", 0.9)])) == ""


def test_text_is_escaped():
    html = render.code_row(code("I10", 0.9, description="<script>alert(1)</script>"))
    assert "<script>" not in html and "&lt;script&gt;" in html
