import json

import pytest

from drone_suru_simulatoru import cli


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_senaryolar_lists_everything(capsys):
    code, out, _ = run(capsys, "senaryolar")
    assert code == 0 and all(name in out for name in ("takas", "surusu", "formasyon", "kapsama", "gorev", "vicsek"))


def test_takas_text_json_and_plot(capsys, tmp_path):
    png = tmp_path / "takas.png"
    code, out, _ = run(capsys, "takas", "--n", 8, "--grafik", png)
    assert code == 0 and "çarpışan çift: 0" in out and "varan: 8/8" in out and png.stat().st_size > 5000
    code, out, _ = run(capsys, "takas", "--n", 8, "--kacinmasiz")
    assert "çarpışan çift: 28" in out
    code, out, _ = run(capsys, "takas", "--n", 6, "--json")
    data = json.loads(out)
    assert data["n"] == 6 and data["carpisan_cift"] == 0 and data["avoid"] is True


def test_surusu_formasyon_gorev_kapsama(capsys):
    code, out, _ = run(capsys, "surusu", "--n", 20, "--engel", 1, "--sure", 40)
    assert code == 0 and "engele en yakın boşluk" in out and "20/20" in out
    code, out, _ = run(capsys, "formasyon", "--n", 6, "--sekil", "cizgi", "--topoloji", "tam", "--kacinma")
    assert code == 0 and "ölçülen" in out and "öngörülen" in out
    code, out, _ = run(capsys, "gorev", "--n", 8, "--hedef", 5, "--yontem", "tam")
    assert code == 0 and "5 hedef" in out and "8/8" in out
    code, out, _ = run(capsys, "kapsama", "--n", 8, "--yogunluk", "duzgun", "--sure", 60, "--ariza", "30:2")
    assert code == 0 and "2 drone arızalı" in out and "altıgen alt sınır" in out


def test_formation_cli_numbers_agree_with_the_scenario(capsys):
    code, out, _ = run(capsys, "formasyon", "--n", 8, "--topoloji", "halka", "--tohum", 3, "--json")
    data = json.loads(out)
    assert abs(data["olculen_hiz"] - data["ongoru_hiz"]) / data["ongoru_hiz"] < 2e-3


def test_vicsek_table_and_plot(capsys, tmp_path):
    png = tmp_path / "v.png"
    code, out, _ = run(capsys, "vicsek", "--n", 60, "--kutu", 4, "--nokta", 3, "--adim", 200, "--grafik", png)
    assert code == 0 and "düzen φ" in out and "rastgele yönler" in out and png.stat().st_size > 5000


def test_animation_file_is_written(tmp_path):
    from drone_suru_simulatoru import render, scenarios
    gif = tmp_path / "a.gif"
    render.animate(scenarios.run_takas(n=4), gif, max_frames=6, dpi=40)
    assert gif.read_bytes()[:3] == b"GIF"


def test_dogrulama_exit_code(capsys, monkeypatch):
    from drone_suru_simulatoru import checks
    ok = checks.Check("a", True, "fine")
    bad = checks.Check("b", False, "broken")
    monkeypatch.setattr(checks, "run_all", lambda: [ok])
    code, out, _ = run(capsys, "dogrulama")
    assert code == 0 and "tüm doğrulamalar geçti" in out
    monkeypatch.setattr(checks, "run_all", lambda: [ok, bad])
    code, out, _ = run(capsys, "dogrulama")
    assert code == 1 and "1 başarısız" in out


@pytest.mark.parametrize("argv, message", [
    (["takas", "--ruzgar", "3,2"], "X:Y"),
    (["kapsama", "--ariza", "40"], "X:Y"),
    (["gorev", "--n", 3, "--hedef", 5], "more targets"),
])
def test_usage_errors_exit_2(capsys, argv, message):
    code, _, err = run(capsys, *argv)
    assert code == 2 and message in err


def test_unknown_command_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["uc"])
    assert exc.value.code == 2


def test_turkish_decimal_comma():
    assert cli.f(3.14159, 2) == "3,14" and cli.f(float("nan")) == "-"
