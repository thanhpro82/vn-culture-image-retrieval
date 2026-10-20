from helpers_xlsx import entity, image, write_book
from retrieval.manifest import build_manifest

E1, E2 = "e1", "e2"


def _build(tmp_path, final_images, recheck_images=None, entities=None, **kw):
    ents = entities or [entity(E1, "Áo bà ba"), entity(E2, "Lễ hội")]
    write_book(tmp_path / "final" / "a.xlsx", ents, final_images, **kw)
    if recheck_images is not None:
        write_book(tmp_path / "recheck" / "b.xlsx", ents, recheck_images)
    return build_manifest(tmp_path)


def test_blank_is_keep_x_is_drop(tmp_path):
    df = _build(tmp_path, [image(E1, "Áo bà ba", "http://x/1.jpg", None),
                           image(E1, "Áo bà ba", "http://x/2.jpg", "x")])
    assert df["url"].tolist() == ["http://x/1.jpg"]


def test_recheck_overrides_final(tmp_path):
    final = [image(E1, "Áo bà ba", "http://x/1.jpg", "x"),
             image(E1, "Áo bà ba", "http://x/2.jpg", None)]
    recheck = [image(E1, "Áo bà ba", "http://x/1.jpg", None),
               image(E1, "Áo bà ba", "http://x/2.jpg", "x")]
    df = _build(tmp_path, final, recheck)
    assert df["url"].tolist() == ["http://x/1.jpg"]


def test_short_rows_are_padded(tmp_path):
    df = _build(tmp_path, [image(E1, "Áo bà ba", "http://x/1.jpg", None)], strip_dimension=True)
    assert len(df) == 1


def test_url_in_two_entities_is_one_row(tmp_path):
    df = _build(tmp_path, [image(E1, "Áo bà ba", "http://x/1.jpg", None),
                           image(E2, "Lễ hội", "http://x/1.jpg", None)])
    assert len(df) == 1
    assert sorted(df.loc[0, "entity_ids"]) == [E1, E2]
    assert sorted(df.loc[0, "entity_names"]) == ["Lễ hội", "Áo bà ba"]


def test_skips_lock_files(tmp_path):
    (tmp_path / "final").mkdir(parents=True)
    (tmp_path / "final" / "~$a.xlsx").write_bytes(b"not a zip")
    df = _build(tmp_path, [image(E1, "Áo bà ba", "http://x/1.jpg", None)])
    assert len(df) == 1


def test_row_has_entity_metadata_and_stable_id(tmp_path):
    df = _build(tmp_path, [image(E1, "Áo bà ba", "http://x/1.jpg", None, caption="một chiếc áo")])
    row = df.iloc[0]
    assert row["caption"] == "một chiếc áo"
    assert row["region"] == "Nam Bộ" and row["branch"] == "B"
    assert len(row["image_id"]) == 16
    assert _build(tmp_path, [image(E1, "Áo bà ba", "http://x/1.jpg", None)]).iloc[0]["image_id"] == row["image_id"]
