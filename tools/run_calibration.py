"""CLI entry point for the autocal calibration engine.

Usage:
    pixi run python tools/run_calibration.py input.mcap output.mcap [options]

Options:
    --sift-features N     Max SIFT features per image (default: 0 = unlimited)
    --match-ratio R       Lowe ratio test threshold (default: 0.75)
    --min-matches N       Minimum matches per pair (default: 8)
    --lm-iterations N     Max LM iterations (default: 50)
    --pose-noise-m M      GPS prior translation noise in metres (default: 2.0)
    --cal-noise-frac F    Fractional noise on initial focal length (default: 0.3)
    --pixel-noise-px P    Reprojection pixel noise (default: 2.0)
"""

import argparse
import sys
import time

from autocal.engine.calibration import CalibrationEngine, CalibrationOptions, discover_topics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run visual SfM calibration on an MCAP file."
    )
    parser.add_argument("input", help="Input MCAP file path")
    parser.add_argument("output", help="Output MCAP file path")
    parser.add_argument("--sift-features", type=int, default=0)
    parser.add_argument("--match-ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--lm-iterations", type=int, default=50)
    parser.add_argument("--pose-noise-m", type=float, default=2.0)
    parser.add_argument("--cal-noise-frac", type=float, default=0.3)
    parser.add_argument("--cal-cx-noise-frac", type=float, default=0.01)
    parser.add_argument("--pixel-noise-px", type=float, default=2.0)
    parser.add_argument("--nadir", action="store_true",
                        help="Initialise cameras looking straight down (UAV nadir survey)")
    parser.add_argument("--max-tracks", type=int, default=2000,
                        help="Max triangulated tracks in GTSAM graph (default: 2000)")
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    # Show what topics are in the input
    topics = discover_topics(args.input)
    print("Topics in input MCAP:")
    for topic, schema in sorted(topics.items()):
        print(f"  {topic:<40} {schema}")
    print()

    opts = CalibrationOptions(
        sift_features=args.sift_features,
        match_ratio=args.match_ratio,
        min_matches=args.min_matches,
        lm_iterations=args.lm_iterations,
        pose_noise_m=args.pose_noise_m,
        cal_noise_frac=args.cal_noise_frac,
        cal_cx_noise_frac=args.cal_cx_noise_frac,
        pixel_noise_px=args.pixel_noise_px,
        nadir_camera=args.nadir,
        max_tracks=args.max_tracks,
    )

    engine = CalibrationEngine(opts)

    t_start = time.time()
    try:
        result = engine.run(args.input, args.output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - t_start

    cal = result["calibration"]
    k = cal.k()
    print(f"Calibration complete in {elapsed:.1f}s")
    print(f"  Tracks triangulated: {result['n_tracks']}")
    print(f"  fx  = {cal.fx():.2f} px")
    print(f"  fy  = {cal.fy():.2f} px")
    print(f"  cx  = {cal.px():.2f} px")
    print(f"  cy  = {cal.py():.2f} px")
    print(f"  k1  = {k[0]:.6f}")
    print(f"  k2  = {k[1]:.6f}")
    print(f"  p1  = {k[2]:.6f}")
    print(f"  p2  = {k[3]:.6f}")
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
