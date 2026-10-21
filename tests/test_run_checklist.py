from eval.run_checklist import grade, group_hits, norm


def test_norm_ignores_case_accents_and_spacing():
    assert norm("  Nhà   Thờ  PHÁT Diệm ") == norm("nha tho phat diem")


def test_group_hits_finds_keyword_in_any_text_and_ignores_accents():
    texts = ["Nhà thờ chính tòa Phát Diệm (Nhà thờ đá)", "Chùa Một Cột"]
    assert group_hits(texts, [["phát diệm"], ["một cột"], ["rối nước"]]) == [True, True, False]
    assert group_hits(texts, [["PHAT DIEM"]]) == [True]


def test_grade_strict_needs_every_group_in_top10_loose_needs_one_group_in_top5():
    top = [f"ảnh {i}" for i in range(10)]
    top[7] = "áo dài đỏ"        # chỉ nằm ở hạng 8
    top[0] = "chùa cổ"
    must = [["áo dài"], ["chùa"]]
    g = grade(top, must)
    assert g["strict"] is True          # đủ hai nhóm trong top 10
    assert g["loose"] is True           # 'chùa' có trong top 5
    top2 = ["ảnh"] * 10
    top2[7] = "áo dài đỏ"
    g2 = grade(top2, must)
    assert g2["strict"] is False and g2["loose"] is False   # thiếu 'chùa'; 'áo dài' chỉ ở hạng 8 (ngoài top 5)


def test_grade_with_no_texts_fails():
    assert grade([], [["a"]]) == {"strict": False, "loose": False, "groups": [False]}
