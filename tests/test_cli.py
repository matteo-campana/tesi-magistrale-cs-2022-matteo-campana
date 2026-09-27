# SPDX-License-Identifier: AGPL-3.0-or-later
import json

from rsseg.cli import build_parser, main
from rsseg.console import _split_per_class, format_value, result_table


def test_json_flag_before_or_after_subcommand():
    parser = build_parser()
    assert parser.parse_args(["--json", "train", "-c", "x.yaml"]).json
    assert parser.parse_args(["train", "-c", "x.yaml", "--json"]).json
    assert not parser.parse_args(["train", "-c", "x.yaml"]).json


def test_backbones_command(capsys):
    assert main(["backbones"]) == 0
    out = capsys.readouterr().out
    for arch in ("resnet18", "swin_t", "vit_b_16"):
        assert arch in out


def test_report_json_output(tmp_path, capsys):
    scores = {"oa": 0.9, "mf1": 0.8, "miou": 0.7}
    (tmp_path / "report.json").write_text(
        json.dumps({"name": "r", "all_classes": scores, "without_clutter": scores}),
        encoding="utf-8",
    )
    assert main(["report", str(tmp_path), "--json"]) == 0
    assert "| r | 1 | 0.9000 |" in capsys.readouterr().out


def test_result_rendering_helpers():
    rest, tables = _split_per_class(
        {"run": "a", "test": {"miou": 0.5, "per_class": {"car": {"iou": 0.1}}}}
    )
    assert rest == {"run": "a", "test": {"miou": 0.5}}
    assert tables == {"test": {"car": {"iou": 0.1}}}
    assert result_table(rest).row_count == 2
    assert format_value(0.12345) == "0.1235"
    assert format_value(1e-6) == "1e-06"
    assert format_value(3) == "3"
