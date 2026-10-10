"""Command line interface: ``dronesuru`` (subcommands are ASCII-transliterated Turkish)."""
from __future__ import annotations

import argparse
import json
import math
import sys

from . import __version__


class UsageError(Exception):
    pass


def f(x, nd=2):
    """Number with a Turkish decimal comma."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}".replace(".", ",")


def pair(text, name):
    try:
        a, b = text.split(":")
        return float(a), float(b)
    except ValueError:
        raise UsageError(f"{name} 'X:Y' biçiminde olmalı (ör. 3:-2), verilen: {text!r}")


def finish(result, args):
    """Optional outputs shared by every scenario command."""
    from . import render
    if args.grafik:
        render.plot_result(result, args.grafik)
        print(f"grafik: {args.grafik}")
    if args.animasyon:
        render.animate(result, args.animasyon)
        print(f"animasyon: {args.animasyon}")


def emit(result, args, text):
    if args.json:
        data = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in result.summary.items()}
        print(json.dumps(data, ensure_ascii=False, indent=2, default=float))
    else:
        print(text)
    finish(result, args)
    return 0


def collisions_line(s):
    return (f"  çarpışan çift: {s['carpisan_cift']} (en küçük merkez aralığı {f(s['en_kucuk_aralik_m'])} m; "
            f"temas 2·yarıçap = 0,60 m)")


# ----------------------------------------------------------------------------------------------
def cmd_senaryolar(args):
    from .scenarios import SCENARIOS
    for name, text in SCENARIOS.items():
        print(f"  {name:10s} {text}")
    print("  vicsek     Vicsek modeli: düzen parametresi - gürültü taraması (ayrı komut)")
    return 0


def cmd_takas(args):
    from . import scenarios
    r = scenarios.run_takas(n=args.n, radius=args.yaricap, duration=args.sure, avoid=not args.kacinmasiz,
                            reciprocal=args.karsilikli, cruise=args.seyir,
                            wind=pair(args.ruzgar, "--ruzgar"), seed=args.tohum)
    s = r.summary
    mode = ("kaçınma yok" if not s["avoid"] else
            "hız engeli, " + ("karşılıklı (RVO)" if s["reciprocal"] else "komşu sabit hızla gidiyor varsayımı"))
    text = "\n".join([
        f"Takas: {s['n']} drone, {f(args.yaricap, 0)} m çemberde karşı noktaya; seyir {f(s['cruise'], 1)} m/s; {mode}",
        collisions_line(s),
        f"  varan: {s['varan']}/{s['n']}, son varış {f(s['varis_son_s'], 1)} s (simülasyon {f(s['sure_s'], 0)} s)",
        f"  yol verimi (doğru çizgi / alınan yol): {f(s['yol_verimi'], 3)}",
        f"  kontrol eforu: {f(s['kontrol_eforu'], 1)} m²/s³",
    ])
    return emit(r, args, text)


def cmd_surusu(args):
    from . import scenarios
    r = scenarios.run_surusu(n=args.n, n_obstacles=args.engel, duration=args.sure, seed=args.tohum,
                             wind=pair(args.ruzgar, "--ruzgar"))
    s = r.summary
    text = "\n".join([
        f"Sürü: {s['n']} drone, {s['engel']} engel, hedef (90, 0), {f(s['sure_s'], 0)} s",
        collisions_line(s),
        f"  engele en yakın boşluk: {f(s['en_yakin_engel_m'])} m",
        f"  hedefin 15 m içinde: {s['hedefe_yakin']}/{s['n']}; hareket sırasında ortalama düzen parametresi "
        f"{f(s['ort_duzen'], 3)}; son küme yarıçapı {f(s['son_kume_yaricapi_m'])} m",
    ])
    return emit(r, args, text)


def cmd_formasyon(args):
    from . import scenarios
    r = scenarios.run_formasyon(n=args.n, shape=args.sekil, topology=args.topoloji, gain=args.kazanc,
                                duration=args.sure, seed=args.tohum, v_ref=pair(args.hiz_ref, "--hiz-ref"),
                                wind=pair(args.ruzgar, "--ruzgar"), avoid=args.kacinma)
    s = r.summary
    text = "\n".join([
        f"Formasyon: {s['n']} drone, şekil {s['sekil']}, iletişim grafı {s['topoloji']} "
        f"(cebirsel bağlantı λ₂ = {f(s['cebirsel_baglanti'], 4)}), kazanç {f(s['kazanc'], 2)}, {f(s['sure_s'], 0)} s",
        f"  hata: {f(s['baslangic_hata_m'], 1)} m -> {s['son_hata_m']:.2e} m (rms)",
        f"  bozunma hızı: ölçülen {f(s['olculen_hiz'], 5)} 1/s, Laplace tayfından öngörülen {f(s['ongoru_hiz'], 5)} 1/s",
        collisions_line(s) + ("" if args.kacinma else "  (kaçınma kapalı: --kacinma ile açılır)"),
    ])
    return emit(r, args, text)


def cmd_kapsama(args):
    from . import scenarios
    kill = None
    if args.ariza:
        t, k = pair(args.ariza, "--ariza")
        kill = (t, int(k))
    r = scenarios.run_kapsama(n=args.n, density=args.yogunluk, duration=args.sure, seed=args.tohum, kill=kill,
                              wind=pair(args.ruzgar, "--ruzgar"), avoid=not args.kacinmasiz)
    s = r.summary
    lines = [
        f"Kapsama: {s['n']} drone, {s['yogunluk']} yoğunluk, 100×100 m, {f(s['sure_s'], 0)} s"
        + (f", {s['yok_edilen']} drone arızalı" if s["yok_edilen"] else ""),
        f"  yerleşim maliyeti H: {s['baslangic_maliyet']:.4g} -> {s['son_maliyet']:.4g}",
        f"  aynı başlangıçtan düz Lloyd yakınsaması: {s['lloyd_sinir_maliyet']:.4g}; "
        f"6 rastgele başlangıcın en iyisi: {s['en_iyi_yeniden_baslatma']:.4g}",
    ]
    if "altigen_alt_sinir" in s:
        lines.append(f"  altıgen alt sınır (Fejes Tóth): {s['altigen_alt_sinir']:.4g}; "
                     f"son maliyet / alt sınır = {f(s['son_maliyet'] / s['altigen_alt_sinir'], 3)}")
    lines.append(collisions_line(s))
    return emit(r, args, "\n".join(lines))


def cmd_gorev(args):
    from . import scenarios
    r = scenarios.run_gorev(n=args.n, n_targets=args.hedef, method=args.yontem, duration=args.sure,
                            seed=args.tohum)
    s = r.summary
    text = "\n".join([
        f"Görev atama: {s['n']} drone, {s['hedef']} hedef, yöntem {s['yontem']}"
        + (f" ({s['ihale_sayisi']} teklif)" if s["ihale_sayisi"] else ""),
        f"  toplam atama mesafesi: {f(s['atama_maliyeti_m'], 1)} m",
        f"  varan: {s['varan']}/{s['n']}, son varış {f(s['varis_son_s'], 1)} s",
        collisions_line(s),
    ])
    return emit(r, args, text)


def cmd_vicsek(args):
    import numpy as np

    from . import render, vicsek
    etas = np.linspace(args.gurultu_min, 2 * np.pi, args.nokta)
    rows = vicsek.sweep(args.n, args.kutu, etas, steps=args.adim, burn_in=args.adim // 2, seed=args.tohum)
    print(f"Vicsek modeli: N = {args.n}, L = {f(args.kutu, 1)}, yoğunluk {f(args.n / args.kutu ** 2, 2)}, "
          f"{args.adim} adım (ilk yarısı atılır)")
    print("     η    düzen φ   std    N·var(φ)")
    for eta, mean, std, chi in rows:
        print(f"  {f(eta, 2):>5s}   {f(mean, 3):>6s}  {f(std, 3):>6s}  {f(chi, 3):>7s}")
    print(f"  rastgele yönler için beklenen φ = √π / (2√N) = {f(vicsek.random_phase_order(args.n), 4)}")
    if args.grafik:
        render.plot_vicsek(rows, args.n, args.kutu, args.grafik)
        print(f"grafik: {args.grafik}")
    return 0


def cmd_dogrulama(args):
    from . import checks
    failed = 0
    for check in checks.run_all():
        print(f"[{'TAMAM' if check.ok else 'HATA '}] {check.name}\n        {check.detail}")
        failed += not check.ok
    print(f"\n{failed} başarısız doğrulama" if failed else "\ntüm doğrulamalar geçti")
    return 1 if failed else 0


# ----------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="dronesuru", description="2B drone sürü simülatörü (gerçek donanım yok)")
    p.add_argument("--version", action="version", version=f"dronesuru {__version__}")
    sub = p.add_subparsers(dest="komut", required=True)

    def common(sp, anim=True):
        sp.add_argument("--tohum", type=int, default=0, help="rastgele sayı tohumu")
        sp.add_argument("--grafik", metavar="PNG", help="yol ve zaman serisi grafiği")
        if anim:
            sp.add_argument("--animasyon", metavar="GIF", help="animasyonlu GIF")
        sp.add_argument("--json", action="store_true", help="özeti JSON olarak yaz")
        sp.add_argument("--ruzgar", default="0:0", metavar="X:Y", help="sabit rüzgar hızı (m/s)")

    sub.add_parser("senaryolar", help="senaryoları listele").set_defaults(func=cmd_senaryolar)

    sp = sub.add_parser("takas", help="çember üzerinde karşı noktaya geçiş (hız engeli kaçınması)")
    sp.add_argument("--n", type=int, default=12)
    sp.add_argument("--yaricap", type=float, default=20.0, help="çember yarıçapı (m)")
    sp.add_argument("--seyir", type=float, default=None, help="seyir hızı (m/s); varsayılan 0,5·v_maks")
    sp.add_argument("--sure", type=float, default=None)
    sp.add_argument("--kacinmasiz", action="store_true", help="kaçınmayı kapat")
    sp.add_argument("--karsilikli", action="store_true", help="karşılıklı hız engeli (RVO)")
    common(sp)
    sp.set_defaults(func=cmd_takas)

    sp = sub.add_parser("surusu", help="Reynolds sürüsü: engeller arasından hedefe")
    sp.add_argument("--n", type=int, default=30)
    sp.add_argument("--engel", type=int, default=2, help="engel sayısı (0-4)")
    sp.add_argument("--sure", type=float, default=60.0)
    common(sp)
    sp.set_defaults(func=cmd_surusu)

    sp = sub.add_parser("formasyon", help="uzlaşma tabanlı formasyon kontrolü")
    sp.add_argument("--n", type=int, default=8)
    sp.add_argument("--sekil", default="cember", choices=["cember", "cizgi", "v", "izgara"])
    sp.add_argument("--topoloji", default="halka", choices=["halka", "yol", "tam", "knn"])
    sp.add_argument("--kazanc", type=float, default=0.8)
    sp.add_argument("--hiz-ref", default="0:0", metavar="X:Y", help="formasyon merkezinin hızı (m/s)")
    sp.add_argument("--sure", type=float, default=None)
    sp.add_argument("--kacinma", action="store_true", help="hız engeli kaçınmasını aç")
    common(sp)
    sp.set_defaults(func=cmd_formasyon)

    sp = sub.add_parser("kapsama", help="Lloyd algoritmasıyla alan kapsama")
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--yogunluk", default="gauss", choices=["duzgun", "gauss"])
    sp.add_argument("--sure", type=float, default=150.0)
    sp.add_argument("--ariza", metavar="ZAMAN:SAYI", help="ZAMAN saniyede SAYI drone düşer")
    sp.add_argument("--kacinmasiz", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_kapsama)

    sp = sub.add_parser("gorev", help="drone-hedef atama (ihale / açgözlü / tam) ve uçuş")
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--hedef", type=int, default=None, help="hedef sayısı (<= n)")
    sp.add_argument("--yontem", default="ihale", choices=["ihale", "acgozlu", "tam"])
    sp.add_argument("--sure", type=float, default=None)
    common(sp)
    sp.set_defaults(func=cmd_gorev)

    sp = sub.add_parser("vicsek", help="Vicsek modeli: düzen parametresi - gürültü")
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--kutu", type=float, default=6.0, help="kutu kenarı L")
    sp.add_argument("--nokta", type=int, default=8, help="gürültü örnek sayısı")
    sp.add_argument("--gurultu-min", type=float, default=0.3)
    sp.add_argument("--adim", type=int, default=1500)
    sp.add_argument("--tohum", type=int, default=0)
    sp.add_argument("--grafik", metavar="PNG")
    sp.set_defaults(func=cmd_vicsek)

    sub.add_parser("dogrulama", help="model doğrulamalarını çalıştır").set_defaults(func=cmd_dogrulama)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (UsageError, ValueError) as exc:
        print(f"hata: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
