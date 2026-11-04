"""
Sylheti thesis dataset -> 16 kHz mono 16-bit WAV
==================================================

Reads mapping.csv (made from the audio-alignment audit) and converts every
recording to a clean, consistently named WAV file:

    wav_16k/<Speaker>/<Speaker>_<ExcelRow>.wav          (one take per sentence)
    wav_16k/<Speaker>/alt/<Speaker>_<ExcelRow>_altN.wav (extra takes, only with --include-alt)

It also writes, inside the output folder:
    metadata.csv           file_name, speaker, excel_row, sylheti, bangla, duration_sec, confidence
                           (Hugging Face "audiofolder" format: load_dataset("audiofolder", data_dir=...))
    conversion_report.csv  every file that failed, is too short (<0.4 s) or very long (>30 s)

Works on Windows, Mac, Linux and Google Colab. Needs Python 3.8+ and ffmpeg.
Safe to stop and re-run: files that already exist are skipped.

Usage (from the "Ultimate Thisis Data" folder):
    python convert_to_wav.py
    python convert_to_wav.py --trim-silence          # also cut silence at start/end
    python convert_to_wav.py --include-alt           # also convert extra takes
    python convert_to_wav.py --speakers A2 A7   # only some speakers
    python convert_to_wav.py --root "/content/drive/MyDrive/Ultimate Thisis Data"   # Colab
"""
import argparse, csv, os, shutil, subprocess, sys, time, wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,"
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,areverse")


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    print("\n[!] ffmpeg was not found.\n"
          "    Windows: open PowerShell and run:  winget install Gyan.FFmpeg\n"
          "             then close and reopen PowerShell / the terminal.\n"
          "    Colab:   ffmpeg is already installed.\n"
          "    Mac:     brew install ffmpeg\n")
    sys.exit(1)


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def convert(ffmpeg, src, dst, trim):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn"]
    if trim:
        cmd += ["-af", TRIM]
    cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(tmp)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 100:
        if tmp.exists():
            tmp.unlink()
        return (p.stderr or "ffmpeg failed").strip().splitlines()[-1:] or ["ffmpeg failed"]
    os.replace(tmp, dst)
    return None


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(here), help='folder that contains "Thesis Data 2" (default: this script\'s folder)')
    ap.add_argument("--mapping", default=None, help="mapping.csv (default: <root>/mapping.csv)")
    ap.add_argument("--out", default=None, help="output folder (default: <root>/wav_16k)")
    ap.add_argument("--include-alt", action="store_true", help="also convert extra takes (status alt_take)")
    ap.add_argument("--trim-silence", action="store_true", help="cut silence at the start and end")
    ap.add_argument("--speakers", nargs="*", help="only these speakers")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    root = Path(a.root)
    mapping = Path(a.mapping) if a.mapping else root / "mapping.csv"
    out = Path(a.out) if a.out else root / "wav_16k"
    audio_root = root / "Thesis Data 2"
    ffmpeg = find_ffmpeg()
    if not mapping.exists():
        sys.exit(f"[!] mapping file not found: {mapping}")
    if not audio_root.exists():
        sys.exit(f'[!] folder not found: {audio_root}\n    Use --root to point at the folder that contains "Thesis Data 2".')

    wanted = {"primary", "second_voice"} | ({"alt_take"} if a.include_alt else set())
    jobs = []
    with open(mapping, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] not in wanted or not r["wav_name"]:
                continue
            if a.speakers and r["speaker"] not in a.speakers:
                continue
            src = audio_root.joinpath(*r["original_file"].replace("\\", "/").split("/"))
            dst = out.joinpath(*r["wav_name"].split("/"))
            jobs.append((r, src, dst))

    print(f"{len(jobs)} files to convert -> {out}   (workers: {a.workers}, trim silence: {a.trim_silence})")
    problems, done, t0 = [], 0, time.time()

    def work(job):
        r, src, dst = job
        if not src.exists():
            return job, "source file not found"
        if dst.exists() and wav_duration(dst):
            return job, None
        err = convert(ffmpeg, src, dst, a.trim_silence)
        return job, (" ".join(err) if err else None)

    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            (r, src, dst), err = fut.result()
            done += 1
            if err:
                problems.append((r["speaker"], r["excel_row"], r["original_file"], "FAILED: " + err))
            if done % 500 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{len(jobs)}  ({rate:.0f} files/s, ~{(len(jobs)-done)/max(rate,1e-6)/60:.0f} min left)")

    # metadata + quality checks
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "speaker", "excel_row", "status", "sylheti", "bangla", "duration_sec", "confidence"])
        for r, src, dst in jobs:
            d = wav_duration(dst) if dst.exists() else None
            if d is None:
                continue
            total += d
            if d < 0.4:
                problems.append((r["speaker"], r["excel_row"], r["original_file"], f"too short ({d:.2f}s)"))
            elif d > 30:
                problems.append((r["speaker"], r["excel_row"], r["original_file"], f"very long ({d:.1f}s)"))
            w.writerow([dst.relative_to(out).as_posix(), r["speaker"], r["excel_row"], r["status"],
                        r["sylheti"], r["bangla"], f"{d:.3f}", r["confidence"]])
    with open(out / "conversion_report.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "excel_row", "original_file", "problem"])
        w.writerows(sorted(problems))
    print(f"\nDone. Total audio: {total/3600:.1f} hours.  Problems: {len(problems)} "
          f"(see {out / 'conversion_report.csv'})")


if __name__ == "__main__":
    main()
