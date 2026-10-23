"""Render the 'profile photos' for PR #61705 straight from the torch-profiler traces.

Usage: python3 make_profile_figs.py [light|dark]
Writes figs/pr61705_{timeline_qwen,timeline_nemotron,kernel_times}_<mode>.png
"""
import glob, gzip, json, os, statistics as st, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MODE = (sys.argv[1] if len(sys.argv) > 1 else "light")
P = {                      # validated categorical slots 1 & 2 + chart chrome
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", other="#c3c2b7"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926", other="#4a4a46"),
}[MODE]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "figure.facecolor": P["surface"], "axes.facecolor": P["surface"],
    "savefig.facecolor": P["surface"], "text.color": P["ink"],
    "axes.labelcolor": P["ink2"], "xtick.color": P["muted"], "ytick.color": P["muted"],
})

VIT = ("FlashAttnFwdSm90", "flash_fwd_sm90FlashAttention")


def kernels(d):
    f = glob.glob(f"{d}/rank0.*.pt.trace.json.gz")[0]
    k = [e for e in json.load(gzip.open(f))["traceEvents"] if e.get("cat") == "kernel"]
    return sorted(k, key=lambda e: e["ts"])


def vit_calls(ks):
    """The 128 encoder-attention kernels = the only name called exactly 128 times."""
    cnt = {}
    for e in ks:
        cnt[e["name"]] = cnt.get(e["name"], 0) + 1
    name = next(n for n, c in cnt.items() if c == 128 and any(v in n for v in VIT))
    return [e for e in ks if e["name"] == name], name


def one_image_window(ks, layers=32, block=1):
    """Time window covering image `block`'s encoder pass (32 attention calls)."""
    v, name = vit_calls(ks)
    sel = v[block * layers:(block + 1) * layers]
    t0, t1 = sel[0]["ts"], sel[-1]["ts"] + sel[-1]["dur"]
    inside = [e for e in ks if e["ts"] >= t0 and e["ts"] < t1]
    return t0, t1, inside, name, sum(e["dur"] for e in sel)


def timeline(model, dmain, dpr, out, note):
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 3.5), sharex=True)
    fig.subplots_adjust(left=0.175, right=0.995, top=0.70, bottom=0.185, hspace=0.30)
    spans = []
    for ax, d, arm, color in ((axes[0], dmain, "main — FA3", P["s1"]),
                              (axes[1], dpr, "this PR — FA4", P["s2"])):
        ks = kernels(d)
        t0, t1, inside, name, attn_us = one_image_window(ks)
        spans.append((t1 - t0) / 1000)
        for e in inside:
            is_vit = any(v in e["name"] for v in VIT) and e["dur"] > 100
            ax.add_patch(Rectangle(((e["ts"] - t0) / 1000, 0.18 if is_vit else 0.33),
                                   max(e["dur"] / 1000, 0.012), 0.64 if is_vit else 0.34,
                                   facecolor=color if is_vit else P["other"],
                                   edgecolor="none"))
        ax.set_ylim(0, 1); ax.set_yticks([])
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(P["axis"])
        ax.grid(axis="x", color=P["grid"], lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.012, 0.88, arm, transform=ax.transAxes, va="top", ha="right",
                fontsize=11, fontweight="bold", color=P["ink"])
        ax.text(-0.012, 0.50, f"attention  {attn_us/1000:.1f} ms", transform=ax.transAxes,
                va="center", ha="right", fontsize=9.5, color=P["ink2"])
        ax.text(-0.012, 0.14, f"whole pass  {(t1-t0)/1000:.1f} ms", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=9.5, color=P["ink2"])
    hi = max(spans) * 1.02
    for ax in axes: ax.set_xlim(0, hi)
    axes[1].set_xlabel("milliseconds on the CUDA stream, from the first attention kernel of the image")
    handles = [Rectangle((0, 0), 1, 1, facecolor=P["s1"]), Rectangle((0, 0), 1, 1, facecolor=P["s2"]),
               Rectangle((0, 0), 1, 1, facecolor=P["other"])]
    axes[0].legend(handles, ["FA3 ViT attention", "FA4 ViT attention", "all other kernels"],
                   loc="lower right", frameon=False, ncol=3, fontsize=9,
                   labelcolor=P["ink2"], bbox_to_anchor=(1.003, 1.12), handlelength=1.6,
                   handleheight=0.9, columnspacing=1.6)
    fig.suptitle(f"{model}: one image's ViT encoder pass on the GPU, H100, bf16",
                 x=0.006, y=0.965, ha="left", fontsize=13, fontweight="bold", color=P["ink"])
    fig.text(0.006, 0.855, note, ha="left", fontsize=9.5, color=P["ink2"])
    fig.savefig(out, dpi=170); plt.close(fig)
    print("wrote", out)


def totals(dirs):
    out = []
    for d in dirs:
        ks = kernels(d)
        v, _ = vit_calls(ks)
        out.append((sum(e["dur"] for e in v) / 1000, sum(e["dur"] for e in ks) / 1000 / 4))
    return out


def summary(out):
    Q_M = ["p3_main_A", "p3_main_B", "p3_main_C"]
    Q_P = ["p3_pr_B", "p3_pr_C", "p3_pr_D"]
    N_M = ["np_prof_main_A", "np_prof_main_B", "np_prof_main_C", "np2_prof_main_A", "np2_prof_main_B"]
    N_P = ["np_prof_pr_A", "np_prof_pr_B", "np2_prof_pr_A", "np2_prof_pr_B"]
    data = {k: totals(v) for k, v in dict(qm=Q_M, qp=Q_P, nm=N_M, np_=N_P).items()}
    groups = [("Qwen2-VL-2B\nViT attention, 4 images", "qm", "qp", 0),
              ("Qwen2-VL-2B\ntotal GPU time, per image", "qm", "qp", 1),
              ("Nemotron-Parse-2.0\nViT attention, 4 images", "nm", "np_", 0),
              ("Nemotron-Parse-2.0\ntotal GPU time, per image", "nm", "np_", 1)]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.34
    for i, (label, a, b, idx) in enumerate(groups):
        va = [r[idx] for r in data[a]]; vb = [r[idx] for r in data[b]]
        for off, vals, col in ((-w/2 - 0.012, va, P["s1"]), (w/2 + 0.012, vb, P["s2"])):
            ax.bar(i + off, st.mean(vals), width=w, color=col, zorder=2,
                   edgecolor=P["surface"], linewidth=2)
            ax.scatter([i + off] * len(vals), vals, s=16, color=P["surface"],
                       edgecolor=P["ink2"], linewidth=1.1, zorder=4)
        ax.text(i - w/2 - 0.012, max(va) + 7, f"{st.mean(va):.1f}", ha="center", va="bottom",
                fontsize=9.5, color=P["ink2"])
        ax.text(i + w/2 + 0.012, max(vb) + 7, f"{st.mean(vb):.1f}", ha="center", va="bottom",
                fontsize=9.5, color=P["ink2"])
        ax.text(i, max(max(va), max(vb)) + 28,
                f"{(st.mean(vb)/st.mean(va)-1)*100:+.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=P["ink"])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5, color=P["ink2"])
    ax.set_ylabel("GPU time (ms)")
    ax.set_ylim(0, 380)
    ax.grid(axis="y", color=P["grid"], lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(P["axis"])
    handles = [Rectangle((0, 0), 1, 1, facecolor=P["s1"]), Rectangle((0, 0), 1, 1, facecolor=P["s2"])]
    ax.legend(handles, ["main (FA3)", "this PR (FA4 above 8192)"], frameon=False,
              loc="upper right", fontsize=10, labelcolor=P["ink2"])
    fig.suptitle("GPU kernel time from the torch profiler, 1x H100 80GB, bf16",
                 x=0.005, y=0.985, ha="left", fontsize=13, fontweight="bold", color=P["ink"])
    fig.text(0.005, 0.925, "bars are the mean of 3 (Qwen2-VL) / 4-5 (Nemotron-Parse) profiled runs per arm; "
                           "dots are the individual runs", ha="left", fontsize=9.5, color=P["ink2"])
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=170); plt.close(fig)
    print("wrote", out)


os.makedirs("figs", exist_ok=True)
timeline("Qwen2-VL-2B at 1792x1792", "p3_main_B", "p3_pr_B",
         f"figs/pr61705_timeline_qwen_{MODE}.png",
         "16,384 ViT patches/image, 32 encoder layers, head_size 80 — every attention call is above the 8192 threshold")
timeline("Nemotron-Parse-2.0 (C-RADIOv2-H)", "np_prof_main_B", "np_prof_pr_B",
         f"figs/pr61705_timeline_nemotron_{MODE}.png",
         "13,320 ViT patches/image, 32 encoder layers, head_size 80 — every attention call is above the 8192 threshold")
summary(f"figs/pr61705_kernel_times_{MODE}.png")
