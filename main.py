"""
Padel Game Analytics — Shot Classification System
Layman AI Internship Assignment
Main entry point
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Padel Game Analytics - Shot Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video input.mp4
  python main.py --video input.mp4 --output output/results
  python main.py --video input.mp4 --show-preview
  python main.py --dashboard
        """
    )
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: output/)")
    parser.add_argument("--show-preview", action="store_true", help="Show live video preview window")
    parser.add_argument("--dashboard", action="store_true", help="Launch analytics dashboard after processing")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame for speed (default: 2)")

    args = parser.parse_args()

    if not args.video and not args.dashboard:
        parser.print_help()
        print("\n[ERROR] Please provide --video <path> to process a video, or --dashboard to view results.")
        sys.exit(1)

    if args.video:
        from analyzer import PadelAnalyzer
        print("=" * 60)
        print("  PADEL GAME ANALYTICS — Shot Classification System")
        print("  Layman AI Internship Assignment")
        print("=" * 60)

        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] Video file not found: {video_path}")
            sys.exit(1)

        analyzer = PadelAnalyzer(
            output_dir=args.output,
            show_preview=args.show_preview,
            conf_threshold=args.conf,
            device=args.device,
            skip_frames=args.skip_frames,
        )
        analyzer.process_video(str(video_path))

    if args.dashboard:
        from dashboard.app import run_dashboard
        run_dashboard(output_dir=args.output)


if __name__ == "__main__":
    main()
