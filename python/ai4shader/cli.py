"""``python -m ai4shader`` -- the offline command-line interface.

Commands
--------
``compilers``
    Which real compilers were found, with resolved paths and version banners.
``validate <file>``
    Run the real compiler for a target.  Exit codes separate "the shader is
    broken" from "no compiler exists" -- see below.
``score <render> <reference>``
    All metrics, optionally including LPIPS and CLIP.
``attribute <render> <reference>``
    The DESIGN.md section 4.3 attribution: concrete ``{node, param, direction}``
    patches, resolved against a document when ``--catalog`` is given.
``batch <doc>...``
    Emit every target for every document through the Node CLI, validate each
    output, and write a diffable JSON report.
``capabilities``
    What can and cannot run here, and why.

Exit codes
----------
=====  ==========================================================================
``0``  success: the assertion the command makes is true
``1``  a real failure: a compiler rejected the source, or the batch has failures
``2``  usage error: bad flags, missing file, unreadable input
``3``  **no verdict exists** -- the required compiler is absent, or ``--tool``
       named a tool that is not installed.  Deliberately distinct from ``1`` so
       CI can tell "the shader is wrong" from "this machine cannot tell".
``4``  ``--require-verified`` was passed and at least one item was unverified
=====  ==========================================================================

``3`` is the one that matters: it makes it impossible for a CI job to mistake an
absent compiler for a passing shader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterable, Sequence

from . import __version__
from . import attribute as attribute_mod
from . import batch as batch_mod
from . import imageio as imageio_mod
from . import score as score_mod
from . import validate as validate_mod

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_NO_VERDICT = 3
EXIT_UNVERIFIED = 4


def _emit(payload: Any, *, as_json: bool, text: str = "") -> None:
    if as_json:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False, sort_keys=False, default=str)
        sys.stdout.write("\n")
    elif text:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


# --------------------------------------------------------------------------- #
# compilers / capabilities
# --------------------------------------------------------------------------- #


def cmd_compilers(args: argparse.Namespace) -> int:
    tools = validate_mod.available(refresh=args.refresh)
    rows = []
    for name, path in tools.items():
        version = validate_mod._tool_version(name, path) if path else None
        rows.append({"tool": name, "path": path, "found": path is not None, "version": version})
    if args.json:
        _emit({"compilers": rows}, as_json=True)
    else:
        for row in rows:
            if row["found"]:
                print(f"  {row['tool']:<20} FOUND  {row['path']}")
                if row["version"]:
                    print(f"  {'':<20}        {row['version']}")
            else:
                print(f"  {row['tool']:<20} absent")
        found = [r["tool"] for r in rows if r["found"]]
        print()
        print(f"  {len(found)} of {len(rows)} present: {', '.join(found) if found else '(none)'}")
        print("  Targets with no compiler present will report 'unverified' -- never 'pass'.")
    return EXIT_OK


def cmd_capabilities(args: argparse.Namespace) -> int:
    caps = score_mod.capabilities(probe=args.probe)
    if args.json:
        _emit(caps, as_json=True)
        return EXIT_OK
    print(f"ai4shader {__version__} -- metric capabilities on this machine")
    print()
    for name in ("ssim", "histogram_distance", "band_energy", "gradient_energy", "color_stats"):
        print(f"  {name:<22} available   ({caps[name]['detail']})")
    for name, label in (("lpips", "lpips"), ("clip_similarity", "clip_similarity")):
        info = caps[name]
        state = "available" if info["available"] else "UNAVAILABLE"
        verified = "verified" if info.get("verified") else ("not yet loaded" if info["available"] else "-")
        print(f"  {name:<22} {state:<12} {verified}")
        print(f"  {'':<22} {info['reason']}")
        for key in ("linear_weights", "backbone_weights", "weights"):
            if info.get(key):
                mark = ""
                if key == "linear_present":
                    mark = ""
                print(f"  {'':<22} {key}: {info[key]}")
        for key in ("linear_present", "backbone_present", "sha256_ok"):
            if key in info and info[key] is not None:
                print(f"  {'':<22} {key}: {info[key]}")
    if not args.probe:
        print()
        print("  Run with --probe to actually load the models and confirm they run offline.")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_mod.validate_file(
        args.path,
        args.target,
        entry=args.entry,
        require_tool=args.tool,
        timeout=args.timeout,
    )
    payload = result.to_dict()
    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"target   : {result.target}")
        print(f"tool     : {result.tool or '-'} {('@ ' + result.tool_path) if result.tool_path else ''}")
        if result.tool_version:
            print(f"version  : {result.tool_version}")
        if result.command:
            print(f"command  : {' '.join(result.command)}")
        for note in result.notes:
            print(f"note     : {note}")
        print(f"verdict  : {'PASS' if result.ok else 'FAIL' if result.ok is False else 'NO VERDICT'}")
        print(f"reason   : {result.reason}")
        if result.diagnostics:
            print(f"messages : {result.error_count} error(s), {result.warning_count} warning(s)")
            for d in result.diagnostics[: args.max_diagnostics]:
                location = f"{d.line}:{d.column}" if d.line is not None else "-"
                print(f"  {location:>10}  {d.severity:<8} {d.message}")
            hidden = len(result.diagnostics) - args.max_diagnostics
            if hidden > 0:
                print(f"  ... {hidden} more (use --max-diagnostics 0 to show all, or --json)")
        elif result.returncode not in (0, None):
            print(f"raw      : exit code {result.returncode}")
            for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
                if text.strip():
                    print(f"  --- {stream} (verbatim) ---")
                    print("  " + text.strip().replace("\n", "\n  "))

    if result.ok is True:
        return EXIT_OK
    if result.ok is False:
        return EXIT_FAIL
    return EXIT_NO_VERDICT


# --------------------------------------------------------------------------- #
# score / attribute
# --------------------------------------------------------------------------- #


def _pad_arg(value: str):
    """``--pad`` accepts the named fills or a numeric grey level."""
    if value in ("mean", "black", "edge"):
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--pad wants 'mean', 'black', 'edge' or a number 0..1 -- got {value!r}"
        ) from exc


def _load_pair(args: argparse.Namespace):
    # `--size 256` parses to (256,); pass the int through so `fit` treats it as a
    # square.  Passing (256,) straight on would make size = ((256,), (256,)).
    size = args.size[0] if len(args.size) == 1 else tuple(args.size)
    render = imageio_mod.load(args.render, size=size, mode=args.fit, pad=args.pad)
    reference = imageio_mod.load(args.reference, size=size, mode=args.fit, pad=args.pad)
    if render.shape != reference.shape:
        raise SystemExit(
            f"error: after fitting, the render is {render.shape} and the reference is "
            f"{reference.shape}; pass --size so both land on the same canvas"
        )
    return render, reference


def cmd_score(args: argparse.Namespace) -> int:
    render, reference = _load_pair(args)
    breakdown = score_mod.score(render, reference, ts_compat=args.ts_compat)
    payload = breakdown.to_dict()
    payload["loss"] = None
    lpips_value = None
    clip_value = None
    if args.lpips:
        lpips_value = score_mod.lpips(render, reference)
        payload["lpips"] = lpips_value
        payload["lpips_available"] = lpips_value is not None
    if args.clip:
        clip_value = score_mod.clip_similarity(render, reference)
        payload["clipSimilarity"] = clip_value
        payload["clip_available"] = clip_value is not None
    payload["loss"] = score_mod.loss(breakdown, lpips_value=lpips_value)

    if args.json:
        _emit(payload, as_json=True)
        return EXIT_OK

    print(f"render    : {args.render}  -> {render.shape[1]}x{render.shape[0]}")
    print(f"reference : {args.reference}")
    print()
    print(f"  ssim                 {breakdown.ssim:8.4f}   (-1..1, 1 = identical, higher better)")
    print(f"  hist                 {breakdown.hist:8.4f}   (0..1, 0 = identical, lower better)")
    if breakdown.ssim_low is not None:
        print(f"  ssimLow (1/4 scale)  {breakdown.ssim_low:8.4f}   (layout vs grain)")
    if breakdown.bands:
        print()
        print(f"  {'band':<6} {'render':>9} {'reference':>10} {'ratio':>8}")
        for band in ("low", "mid", "high"):
            r = breakdown.bands.get(band, 0.0)
            ref = breakdown.bands_ref.get(band, 0.0)
            ratio = score_mod.band_ratio(breakdown.bands, breakdown.bands_ref, band)
            shown = "inf" if ratio == float("inf") else f"{ratio:.3f}"
            print(f"  {band:<6} {r:9.5f} {ref:10.5f} {shown:>8}")
    if breakdown.luma_offset is not None:
        print()
        print(f"  mean luma            {breakdown.mean_luma['cur']:8.4f} vs {breakdown.mean_luma['ref']:.4f}"
              f"   offset {breakdown.luma_offset:+.4f}")
    if breakdown.saturation:
        print(f"  saturation           {breakdown.saturation['cur']:8.4f} vs {breakdown.saturation['ref']:.4f}")
    if breakdown.gradient:
        print(f"  gradient RMS         {breakdown.gradient['cur']:8.4f} vs {breakdown.gradient['ref']:.4f}")
    if args.lpips or args.clip:
        print()
        if args.lpips:
            print(f"  lpips                {lpips_value if lpips_value is not None else 'None (weights unavailable)'}")
        if args.clip:
            print(f"  clip similarity      {clip_value if clip_value is not None else 'None (weights unavailable)'}")
    print()
    print(f"  loss (lower better)  {payload['loss']:.4f}")
    if args.ts_compat:
        print()
        print("  NOTE: --ts-compat reproduces score.ts's buggy `high` band. Do not use for decisions.")
    return EXIT_OK


def cmd_attribute(args: argparse.Namespace) -> int:
    render, reference = _load_pair(args)
    breakdown = score_mod.score(render, reference, ts_compat=args.ts_compat)

    catalog = None
    if args.catalog:
        with open(args.catalog, "r", encoding="utf-8-sig") as fh:
            catalog = attribute_mod.catalog_from_document(json.load(fh))

    temporal = None
    if args.temporal:
        frame = imageio_mod.load(args.temporal, size=render.shape[1::-1], mode=args.fit, pad=args.pad)
        temporal = score_mod.ssim(frame, render)

    hints = attribute_mod.attribute(breakdown, catalog=catalog, temporal_ssim=temporal, max_hints=args.max_hints)
    payload = {
        "breakdown": breakdown.to_dict(),
        "verdict": attribute_mod.summarize(breakdown, hints),
        "hints": hints,
        "temporalSsim": temporal,
        "catalog": os.path.basename(args.catalog) if args.catalog else None,
    }
    if args.json:
        _emit(payload, as_json=True)
        return EXIT_OK

    print(attribute_mod.summarize(breakdown, hints))
    print()
    if not hints:
        print("  No attribution rule fired. That means the seven observables in DESIGN.md 4.3 are all")
        print("  within tolerance -- it does NOT by itself mean the render is good.")
        return EXIT_OK
    for i, hint in enumerate(hints, 1):
        target = f"{hint['node']}.{hint['param']}" if hint["node"] else f"(unresolved).{hint['param']}"
        print(f"  {i}. {hint['rule']}  (confidence {hint['confidence']:.2f})")
        print(f"     target    : {target}")
        if hint.get("uniform"):
            print(f"     uniform   : {hint['uniform']}   (e.g. --set {hint['uniform']}={hint.get('suggest')})")
        print(f"     direction : {hint['direction'] or 'undetermined -- explore this parameter'}")
        if hint.get("suggest") is not None:
            print(f"     suggest   : {hint['suggest']}")
        print(f"     why       : {hint['why']}")
        print()
    return EXIT_OK


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #


def cmd_batch(args: argparse.Namespace) -> int:
    targets = batch_mod.TARGETS if args.targets == ["all"] else args.targets
    report = batch_mod.run_matrix(
        args.docs,
        targets,
        out_dir=args.out_dir,
        timeout=args.timeout,
        require_tool=args.tool,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(report.to_json())
    if args.json:
        _emit({"report": report.to_dict(), "written": args.out}, as_json=True)
    else:
        print(report.render_text())
        if args.out:
            print(f"\nreport written to {args.out}")

    summary = report.summary
    if summary.get("fail"):
        return EXIT_FAIL
    if summary.get("cli-missing") or summary.get("emit-error"):
        return EXIT_FAIL
    if summary.get("pass") == 0 and summary.get("unverified"):
        # Nothing was actually verified.  Saying "success" here is exactly the
        # false green this project is built to avoid.
        return EXIT_NO_VERDICT if not args.require_verified else EXIT_UNVERIFIED
    if args.require_verified and summary.get("unverified"):
        return EXIT_UNVERIFIED
    return EXIT_OK


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def _size_arg(value: str) -> tuple[int, ...]:
    try:
        parts = [int(p) for p in value.replace("x", ",").split(",") if p.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--size wants N or W,H -- got {value!r}") from exc
    if len(parts) not in (1, 2) or any(p <= 0 for p in parts):
        raise argparse.ArgumentTypeError(f"--size wants N or W,H with positive values -- got {value!r}")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4shader",
        description="AI4Shader offline layer: real compiler validation, image scoring and failure attribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 success\n"
            "  1 a compiler rejected the source, or the batch has failures\n"
            "  2 usage error\n"
            "  3 NO VERDICT EXISTS (compiler absent, or --tool not installed)\n"
            "  4 --require-verified was given and something was unverified\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"ai4shader {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compilers", help="which real shader compilers were found")
    p.add_argument("--json", action="store_true")
    p.add_argument("--refresh", action="store_true", help="re-probe instead of using the cache")
    p.set_defaults(func=cmd_compilers)

    p = sub.add_parser("capabilities", help="which metrics can run, and why not")
    p.add_argument("--json", action="store_true")
    p.add_argument("--probe", action="store_true", help="actually load LPIPS/CLIP (seconds, ~600 MB)")
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("validate", help="compile one emitted shader with a real compiler")
    p.add_argument("path", help="path to a .frag/.wgsl/.hlsl/.metal/.shader file")
    p.add_argument("--target", choices=sorted(validate_mod.TOOL_FOR_TARGET), default=None)
    p.add_argument("--entry", default=None, help="entry point (default: discovered from the source)")
    p.add_argument("--tool", default=None, help="REQUIRE this specific tool, e.g. --tool dxc")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-diagnostics", type=int, default=40, help="0 = show all")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    def add_image_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--size", type=_size_arg, default=(512,),
                       help="fit both images to N or W,H (default 512). Letterboxed, never stretched.")
        p.add_argument("--fit", choices=("letterbox", "crop", "stretch"), default="letterbox",
                       help="'stretch' corrupts every metric; it exists to reproduce the mistake")
        p.add_argument("--pad", type=_pad_arg, default="mean",
                       help="letterbox fill: mean (default), black, edge, or a float 0..1")
        p.add_argument("--ts-compat", action="store_true",
                       help="reproduce score.ts's buggy high band (cross-checking only)")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("score", help="compute SSIM / histogram / band metrics for a render vs a reference")
    p.add_argument("render")
    p.add_argument("reference")
    add_image_args(p)
    p.add_argument("--lpips", action="store_true", help="include the real LPIPS distance")
    p.add_argument("--clip", action="store_true", help="include CLIP embedding cosine similarity")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("attribute", help="turn a score into concrete {node, param, direction} patches")
    p.add_argument("render")
    p.add_argument("reference")
    add_image_args(p)
    p.add_argument("--catalog", default=None, help="a .shader.json, to resolve real node ids and uniforms")
    p.add_argument("--temporal", default=None, help="a second frame of the SAME render, for the animation rule")
    p.add_argument("--max-hints", type=int, default=4)
    p.set_defaults(func=cmd_attribute)

    p = sub.add_parser("batch", help="emit every target for every document, then validate each output")
    p.add_argument("docs", nargs="+")
    p.add_argument("--targets", nargs="+", default=["all"],
                   help=f"any of {list(batch_mod.TARGETS)}, or 'all' (default)")
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--out-dir", default=None, help="where the emitter writes (default: a temp dir)")
    p.add_argument("--tool", default=None, help="REQUIRE this specific compiler for every target")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--require-verified", action="store_true",
                   help="exit 4 if anything is unverified, instead of exit 0")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_batch)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
