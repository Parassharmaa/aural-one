"""Render the public descriptive audio-evaluation chart from frozen aggregates."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "assets/audio_eval_chart_data.json").read_text())
OUT = ROOT / "assets/audio-evaluation.png"

BG = "#F8FAF8"
INK = "#18333B"
MUTED = "#6A7B80"
TEAL = "#087F73"
REF = "#9AA9AB"
LINE = "#D8E3DF"


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.edgecolor": LINE, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED})
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    title = fig.add_axes((0.055, 0.865, 0.89, 0.115), frameon=False)
    title.axis("off")
    title.text(0, 0.92, "AURAL ONE  /  E2B PREVIEW", color=TEAL, fontsize=12,
               fontweight="bold", va="top")
    title.text(0, 0.64, "Audio evaluation", fontsize=30, fontweight="bold", va="top")
    title.text(0, 0.16, "One native-audio model · speech emotion and audio questions",
               fontsize=14, color=MUTED, va="top")

    left = fig.add_axes((0.055, 0.365, 0.89, 0.44), facecolor=BG)
    left.set_xlim(0, 0.62)
    left.set_ylim(-0.6, 4.65)
    left.set_yticks([])
    left.set_xticks([0, .1, .2, .3, .4])
    left.set_xticklabels(["0", "10", "20", "30", "40%"])
    left.grid(axis="x", color=LINE, linewidth=0.8)
    left.set_axisbelow(True)
    for spine in left.spines.values():
        spine.set_visible(False)
    left.text(0, 4.55, "SPEECH EMOTION", fontsize=13, fontweight="bold", color=INK)
    left.text(0.62, 4.55, "MACRO-F1", fontsize=12, fontweight="bold", color=MUTED,
              ha="right")
    left.scatter([.34], [4.57], s=70, color=TEAL, clip_on=False)
    left.text(.355, 4.57, "Aural One", fontsize=11, color=MUTED, va="center")
    left.scatter([.47], [4.57], s=70, color=REF, clip_on=False)
    left.text(.485, 4.57, "Reference", fontsize=11, color=MUTED, va="center")
    for index, item in enumerate(DATA["emotion_macro_f1"]):
        y = 3.8 - index * .88
        left.text(0, y + .22, item["label"], fontsize=13, fontweight="bold")
        left.text(0, y + .02, item["scope"], fontsize=10.5, color=MUTED)
        ref = item["reference"]
        current = item["aural_one"]
        if ref is not None:
            left.plot([ref, current], [y - .23, y - .23], color=LINE, linewidth=4,
                      solid_capstyle="round", zorder=2)
            left.scatter([ref], [y - .23], s=90, color=REF, zorder=3)
        left.scatter([current], [y - .23], s=125, color=TEAL, zorder=4)
        comparison = (f"{ref * 100:.1f} → {current * 100:.1f}%" if ref is not None
                      else f"{current * 100:.1f}%")
        left.text(.62, y - .23, comparison, ha="right", va="center", fontsize=12,
                  color=INK, fontweight="bold")

    right = fig.add_axes((0.055, 0.095, 0.89, 0.22), facecolor=BG)
    right.set_xlim(0, 1.17)
    right.set_ylim(-0.5, 2.55)
    right.set_yticks([])
    right.set_xticks([0, .25, .5, .75, 1.0])
    right.set_xticklabels(["0", "25", "50", "75", "100%"])
    right.grid(axis="x", color=LINE, linewidth=.8)
    right.set_axisbelow(True)
    for spine in right.spines.values():
        spine.set_visible(False)
    right.text(0, 2.42, "AUDIO QUESTION CHECKS", fontsize=13, fontweight="bold")
    right.text(1.17, 2.42, "CORRECT / TOTAL", fontsize=12, fontweight="bold",
               color=MUTED, ha="right")
    for index, item in enumerate(DATA["audio_question_accuracy"]):
        y = 1.7 - index * .78
        value = item["correct"] / item["total"]
        right.text(0, y + .20, item["label"], fontsize=12, fontweight="bold")
        right.text(0, y + .02, item["scope"], fontsize=10.5, color=MUTED)
        right.barh(y - .21, 1.0, height=.10, color=LINE, edgecolor="none")
        right.barh(y - .21, value, height=.10, color=TEAL, edgecolor="none")
        right.text(1.17, y - .21, f"{item['correct']}/{item['total']} · {value:.0%}",
                   ha="right", va="center", fontsize=11.5, fontweight="bold")

    fig.text(.055, .025,
             "Reference = frozen step-420 checkpoint. Different datasets; sample counts are clips or questions. "
             "See EVALUATION.md for splits and controls.",
             fontsize=9.5, color=MUTED)
    fig.savefig(OUT, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=.25)
    print(OUT)


if __name__ == "__main__":
    main()
