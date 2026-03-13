import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils

# CMS CVD-friendly color palette (Petroff, adopted by CMS 2024)
CMS_COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
colors = [CMS_COLORS[0]]


def set_cms_style():
    """Apply CMS plotting style with CVD-friendly Petroff palette."""
    mpl.rcParams.update({
        # Font
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
        "font.family": "sans-serif",
        "font.size": 14,
        # Axes
        "axes.labelsize": 16,
        "axes.linewidth": 1.5,
        "axes.prop_cycle": mpl.cycler(color=CMS_COLORS),
        # Ticks — inward, visible on all sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.major.size": 8,
        "ytick.minor.size": 4,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 13,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.5,
    })


def cms_label(ax, label="Simulation", loc="left"):
    """Add a CMS-style label (bold 'CMS' + italic qualifier) to the axes."""
    ax.text(0.0, 1.01, r"$\mathbf{CMS}$", fontsize=16,
            fontstyle="normal", transform=ax.transAxes, ha="left", va="bottom")
    ax.text(0.095, 1.01, r"$\it{%s}$" % label, fontsize=13,
            transform=ax.transAxes, ha="left", va="bottom")


set_cms_style()


def dup(a):
    return np.append(a, a[-1])

def make_hist(
    reference,
    generated,
    xlabel="",
    ylabel="a.u.",
    logy=False,
    binning=None,
    label_loc="best",
    model_name="Model",
    fname="",
    leg_font=16,
):

    if binning is None: # default: 50 bins between min and max of reference, we have internal discussion and decided to go with reference binning only, so that the binning is fixed for all participants! 
        lower_bound = np.quantile(reference, 0.0) - 1e-8
        upper_bound = np.quantile(reference, 1.0) + 1e-8
        if "occupancy" in xlabel.lower():
            # avoid binning effects for discrete features
            delta = max(np.ceil((upper_bound - lower_bound) / 50), 1.0)
            binning = np.arange(lower_bound, upper_bound + delta, delta)
            if len(binning) < 2:
                binning = np.linspace(lower_bound, lower_bound + delta, 3)
        else:
            binning = np.linspace(lower_bound, upper_bound, 50)

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(5, 4.5),
        gridspec_kw={"hspace": 0.0, "height_ratios": (3, 1)},
        sharex=True,
    )

    # Geant4 lines
    dist_ref, binning = np.histogram(
        reference,
        bins=binning,
        density=False,
    )
    bin_widths = np.diff(binning)
    dist_ref_normalized = dist_ref / (bin_widths * dist_ref.sum())
    dist_ref_error = dist_ref_normalized / np.sqrt(dist_ref)
    dist_ref_ratio_error = dist_ref_error / dist_ref_normalized
    dist_ref_ratio_error_isnan = np.isnan(dist_ref_ratio_error)
    dist_ref_ratio_error[dist_ref_ratio_error_isnan] = 0.0

    ax[0].step(
        binning,
        dup(dist_ref_normalized),
        label="Geant4",
        linestyle="-",
        alpha=0.8,
        linewidth=1.0,
        color="k",
        where="post",
    )
    ax[0].fill_between(
        binning,
        dup(dist_ref_normalized + dist_ref_error),
        dup(dist_ref_normalized - dist_ref_error),
        step="post",
        color="k",
        alpha=0.2,
    )
    ax[1].fill_between(
        binning,
        dup(1 - dist_ref_ratio_error),
        dup(1 + dist_ref_ratio_error),
        step="post",
        color="k",
        alpha=0.2,
    )

    # Generator lines
    dist_gen, binning = np.histogram(generated, bins=binning, density=False)
    bin_widths = np.diff(binning)
    dist_gen_normalized = dist_gen / (bin_widths * dist_gen.sum())
    dist_gen_error = dist_gen_normalized / np.sqrt(dist_gen)
    ratio = dist_gen_normalized / dist_ref_normalized
    ratio_err = dist_gen_error / dist_ref_normalized
    ratio_isnan = np.isnan(ratio)
    ratio[ratio_isnan] = 1.0
    ratio_err[ratio_isnan] = 0.0

    ax[0].step(
        binning,
        dup(dist_gen_normalized),
        label=model_name,
        where="post",
        linewidth=1.0,
        alpha=1.0,
        color=colors[0],
        linestyle="-",
    )
    ax[0].fill_between(
        binning,
        dup(dist_gen_normalized + dist_gen_error),
        dup(dist_gen_normalized - dist_gen_error),
        step="post",
        color=colors[0],
        alpha=0.2,
    )
    ax[1].step(
        binning,
        dup(ratio),
        linewidth=1.0,
        alpha=1.0,
        color=colors[0],
        where="post",
    )
    ax[1].fill_between(
        binning,
        dup(ratio - ratio_err),
        dup(ratio + ratio_err),
        step="post",
        color=colors[0],
        alpha=0.2,
    )
    ax[1].hlines(
        1.0,
        binning[0],
        binning[-1],
        linewidth=1.0,
        alpha=0.8,
        linestyle="-",
        color="k",
    )
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(binning[0], binning[-1])

    if logy:
        ax[0].set_yscale("log")
    else:
        ax[0].set_ylim(0.0, None)
    ax[1].axhline(0.7, c="k", ls="--", lw=0.5)
    ax[1].axhline(1.3, c="k", ls="--", lw=0.5)
    ax[0].set_ylabel(ylabel, fontsize=leg_font)
    ax[1].set_xlabel(xlabel, fontsize=leg_font)
    ax[1].set_ylabel(r"$\frac{\text{%s}}{\text{Geant4}}$" % model_name, fontsize=leg_font)
    ax[0].legend(
        loc=label_loc,
        frameon=False,
        handlelength=1.2,
        title_fontsize=leg_font,
        fontsize=leg_font,
    )
    cms_label(ax[0])
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
    sep_power = utils._separation_power(
        dist_ref_normalized, dist_gen_normalized, binning
    )

    if len(fname) > 0:
        fig.savefig(fname + ".png")
        fig.savefig(fname + ".pdf", dpi=300, format="pdf")

        #save hists for later
        np.savez(
             fname+".npz", 
             dist_ref=dist_ref_normalized, 
             dist_ref_err=dist_ref_error, 
             dist_gen=dist_gen_normalized, 
             dist_gen_err=dist_gen_error, 
             binning=binning
        )
    plt.close(fig)
    return sep_power


def make_profile(
    ref_profiles,
    gen_profiles,
    xlabel="Layer",
    ylabel="Energy fraction",
    model_name="Model",
    fname="",
    leg_font=16,
    logy=False,
):
    """Plot average profile (longitudinal or transverse).

    Upper panel: mean line with two layers of uncertainty:
      - light shaded band: ±1 std (shower-to-shower spread)
      - darker shaded band: ±1 SEM (statistical uncertainty on the mean)
    Lower panel: ratio with ±1 SEM error band.

    ref_profiles, gen_profiles: arrays of shape (nShowers, nBins) where each
    column is a layer or ring feature value.
    """
    n_bins = ref_profiles.shape[1]
    x = np.arange(n_bins)

    n_ref = ref_profiles.shape[0]
    n_gen = gen_profiles.shape[0]

    ref_mean = np.mean(ref_profiles, axis=0)
    ref_std = np.std(ref_profiles, axis=0)
    ref_sem = ref_std / np.sqrt(n_ref)
    gen_mean = np.mean(gen_profiles, axis=0)
    gen_std = np.std(gen_profiles, axis=0)
    gen_sem = gen_std / np.sqrt(n_gen)

    fig, ax = plt.subplots(
        2, 1, figsize=(6, 4.5),
        gridspec_kw={"hspace": 0.0, "height_ratios": (3, 1)},
        sharex=True,
    )

    # Reference (Geant4) — std band (light) + SEM band (darker)
    ax[0].step(x, ref_mean, where="mid", color="k", linewidth=1.0, alpha=0.8, label="Geant4")
    ax[0].fill_between(x, ref_mean - ref_std, ref_mean + ref_std, alpha=0.1, color="k", step="mid")
    ax[0].fill_between(x, ref_mean - ref_sem, ref_mean + ref_sem, alpha=0.3, color="k", step="mid")

    # Generated — std band (light) + SEM band (darker)
    ax[0].step(x, gen_mean, where="mid", color=colors[0], linewidth=1.0, alpha=1.0, label=model_name)
    ax[0].fill_between(x, gen_mean - gen_std, gen_mean + gen_std, alpha=0.1, color=colors[0], step="mid")
    ax[0].fill_between(x, gen_mean - gen_sem, gen_mean + gen_sem, alpha=0.3, color=colors[0], step="mid")

    # Ratio panel — SEM error bands
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref_mean > 0, gen_mean / ref_mean, 1.0)
        ratio_err = np.where(ref_mean > 0, gen_sem / ref_mean, 0.0)
        ref_ratio_err = np.where(ref_mean > 0, ref_sem / ref_mean, 0.0)

    ax[1].fill_between(x, 1 - ref_ratio_err, 1 + ref_ratio_err, alpha=0.2, color="k", step="mid")
    ax[1].step(x, ratio, where="mid", color=colors[0], linewidth=1.0)
    ax[1].fill_between(x, ratio - ratio_err, ratio + ratio_err, alpha=0.2, color=colors[0], step="mid")
    ax[1].axhline(1.0, color="k", linewidth=1.0, alpha=0.8)
    ax[1].axhline(0.7, c="k", ls="--", lw=0.5)
    ax[1].axhline(1.3, c="k", ls="--", lw=0.5)
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)

    if logy:
        ax[0].set_yscale("log")
    else:
        ax[0].set_ylim(0.0, None)
    ax[0].set_ylabel(ylabel, fontsize=leg_font)
    ax[1].set_xlabel(xlabel, fontsize=leg_font)
    ax[1].set_ylabel(r"$\frac{\text{%s}}{\text{Geant4}}$" % model_name, fontsize=leg_font)
    ax[0].legend(loc="best", frameon=False, fontsize=leg_font)
    cms_label(ax[0])
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

    if len(fname) > 0:
        fig.savefig(fname + ".png")
        fig.savefig(fname + ".pdf", dpi=300, format="pdf")
        np.savez(
            fname + ".npz",
            ref_mean=ref_mean, ref_std=ref_std, ref_sem=ref_sem,
            gen_mean=gen_mean, gen_std=gen_std, gen_sem=gen_sem,
        )
    plt.close(fig)
    return fig
