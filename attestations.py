import numpy as np
import weightedstats as ws
import pathlib
import collections
import json
import matplotlib
import matplotlib.pyplot as plt

ATTESTATION_DIR = "./attestations/"
HEADER_DIR = "./headers/"

MERGE_SLOT = 4700013

colors_darker = ["#16537e", "#bc5800"]


def hex_to_bitfield(bits_hex):
    bits_bytes = bytes.fromhex(bits_hex)
    bits_uint8 = np.frombuffer(bits_bytes, dtype="uint8")
    bits = np.unpackbits(bits_uint8, bitorder="little")
    last_set_bit = np.nonzero(bits)[0][-1]
    return bits[:last_set_bit]


def get_agg_bits(attestation):
    bits_hex = attestation["aggregation_bits"].removeprefix("0x")
    return hex_to_bitfield(bits_hex)


def dict_to_xy(d):
    sorted_items = sorted(d.items())
    x, y = zip(*sorted_items)
    return np.array(x), np.array(y)


def load_attestations(epochs=None):
    slots = None
    if epochs is not None:
        slots = set()
        for epoch in epochs:
            slots = slots.union(set(range(epoch * 32, (epoch + 1) * 32)))

    attestations = {}
    for path in pathlib.Path(ATTESTATION_DIR).iterdir():
        if not path.is_file():
            continue
        slot = int(path.stem)

        if slots and slot not in slots:
            continue

        with open(path) as f:
            a = json.load(f)
            if a is not None:
                if a["execution_optimistic"]:
                    raise ValueError(f"attestations for slot {slot} are optimistic")
                attestations[slot] = a["data"]
            else:
                attestations[slot] = None
    sorted_attestations = collections.OrderedDict(sorted(attestations.items()))
    return sorted_attestations


def load_headers(epochs=None):
    slots = None
    if epochs is not None:
        slots = set()
        for epoch in epochs:
            slots = slots.union(set(range(epoch * 32, (epoch + 1) * 32)))

    headers = {}
    for path in pathlib.Path(HEADER_DIR).iterdir():
        if not path.is_file():
            continue
        slot = int(path.stem)

        if slots and slot not in slots:
            continue

        with open(path) as f:
            h = json.load(f)
            if h is not None:
                if h["execution_optimistic"]:
                    raise ValueError(f"attestations for slot {slot} are optimistic")
                headers[slot] = h["data"]
            else:
                headers[slot] = None
    sorted_headers = collections.OrderedDict(sorted(headers.items()))
    return sorted_headers


def calc_throughput(attestations):
    res = collections.defaultdict(int)

    voted_bitfields = collections.defaultdict(dict)  # slot to index to bitfield
    for incl_slot, atts in attestations.items():
        res[incl_slot] = 0
        for att in atts or []:
            index = int(att["data"]["index"])
            slot = int(att["data"]["slot"])
            voters = get_agg_bits(att)
            if index not in voted_bitfields[slot]:
                voted_bitfields[slot][index] = np.array(
                    [0] * len(voters), dtype="uint8"
                )
            new_voters = voters & ~voted_bitfields[slot][index]
            res[incl_slot] += np.sum(new_voters)
            voted_bitfields[slot][index] |= voters

    return dict_to_xy(res)


def calc_delays(attestations):
    delays = {}
    voted_bitfields = collections.defaultdict(dict)  # slot to index to bitfield
    for incl_slot, atts in attestations.items():
        if not atts:
            delays[incl_slot] = None
            continue
        delays[incl_slot] = []
        for att in atts:
            index = int(att["data"]["index"])
            slot = int(att["data"]["slot"])
            voters = get_agg_bits(att)
            if index not in voted_bitfields[slot]:
                voted_bitfields[slot][index] = np.array(
                    [0] * len(voters), dtype="uint8"
                )
            new_voters = voters & ~voted_bitfields[slot][index]
            delays[incl_slot].append((np.sum(new_voters), incl_slot - slot))
            voted_bitfields[slot][index] |= voters
    return delays


def calc_aggregation_steps(attestations):
    grouped_atts = {}  # (slot, comm index) to att_key to atts
    for incl_slot, atts in attestations.items():
        for att in atts or []:
            att["incl_slot"] = incl_slot
            index = int(att["data"]["index"])
            slot = int(att["data"]["slot"])
            key = (
                att["data"]["beacon_block_root"],
                att["data"]["source"]["epoch"],
                att["data"]["source"]["root"],
                att["data"]["target"]["epoch"],
                att["data"]["target"]["root"],
            )

            if (slot, index) not in grouped_atts:
                grouped_atts[(slot, index)] = {}
            if key not in grouped_atts[(slot, index)]:
                grouped_atts[(slot, index)][key] = []

            grouped_atts[(slot, index)][key].append(att)

    agg_steps = {}
    for (slot, index), d in grouped_atts.items():
        if (slot, index) not in agg_steps:
            agg_steps[(slot, index)] = {}
        for att_key, atts in d.items():
            agg_steps[(slot, index)][att_key] = get_agg_steps(atts)

    return agg_steps


def get_agg_steps(atts):
    if len(atts) == 0:
        return []

    agg_steps = []
    agg_bits = [get_agg_bits(att) for att in atts]
    incl_slots = [att["incl_slot"] for att in atts]
    cum_bits = np.repeat(0, len(agg_bits[0]))

    while len(agg_bits) > 0:
        new_bits = [bits & ~cum_bits for bits in agg_bits]
        i = np.argmax([np.sum(bits) for bits in new_bits])
        bits = agg_bits.pop(int(i))
        incl_slot = incl_slots.pop(i)

        cum_bits |= bits
        agg_steps.append(
            {
                "bits": bits,
                "cum_bits": cum_bits,
                "new_bits": new_bits[i],
                "incl_slot": incl_slot,
            }
        )

    return agg_steps


def calc_disagreements(attestations):
    disagreements = {}  # slot: (roots, sources, targets)
    weights = collections.defaultdict(int)

    atts_by_slot = collections.defaultdict(list)
    for _, atts in attestations.items():
        for att in atts or []:
            atts_by_slot[int(att["data"]["slot"])].append(att)

    voted_bitfields = collections.defaultdict(dict)  # slot to index to bitfield
    for slot, atts in atts_by_slot.items():
        roots = collections.defaultdict(int)
        sources = collections.defaultdict(int)
        targets = collections.defaultdict(int)
        for att in atts or []:
            index = int(att["data"]["index"])
            assert slot == int(att["data"]["slot"])
            voters = get_agg_bits(att)
            if index not in voted_bitfields[slot]:
                voted_bitfields[slot][index] = np.array(
                    [0] * len(voters), dtype="uint8"
                )
            new_voters = voters & ~voted_bitfields[slot][index]
            num_new_voters = np.sum(new_voters)

            roots[att["data"]["beacon_block_root"]] += num_new_voters
            sources[
                (att["data"]["source"]["epoch"], att["data"]["source"]["root"])
            ] += num_new_voters
            targets[
                (att["data"]["source"]["epoch"], att["data"]["source"]["root"])
            ] += num_new_voters
            weights[slot] += num_new_voters

            voted_bitfields[slot][index] |= voters

        if weights[slot] > 0:
            disagreement_tuple = [0.0, 0.0, 0.0]
            for i, d in enumerate([roots, sources, targets]):
                total_vote = sum(d.values())
                _, majority_vote = max(d.items(), key=lambda item: item[1])
                disagreement_tuple[i] = (total_vote - majority_vote) / total_vote
            disagreements[slot] = tuple(disagreement_tuple)
        else:
            disagreements[slot] = (0.0, 0.0, 0.0)

    slots, dis = dict_to_xy(disagreements)
    slots_weights, w = dict_to_xy(weights)
    assert np.all(slots == slots_weights)
    return slots, w, np.transpose(dis)


def calc_reorgs(attestations, headers):
    canonical_roots = set(
        header["root"] for header in headers.values() if header is not None
    )

    weights = collections.defaultdict(int)
    reorgs = {}

    atts_by_slot = collections.defaultdict(list)
    for _, atts in attestations.items():
        for att in atts or []:
            atts_by_slot[int(att["data"]["slot"])].append(att)
            if att["data"]["beacon_block_root"] not in canonical_roots:
                print(att)

    voted_bitfields = collections.defaultdict(dict)  # slot to index to bitfield
    for slot, atts in atts_by_slot.items():
        roots = collections.defaultdict(int)
        for att in atts or []:
            index = int(att["data"]["index"])
            assert slot == int(att["data"]["slot"])
            voters = get_agg_bits(att)
            if index not in voted_bitfields[slot]:
                voted_bitfields[slot][index] = np.array(
                    [0] * len(voters), dtype="uint8"
                )
            new_voters = voters & ~voted_bitfields[slot][index]
            num_new_voters = np.sum(new_voters)

            roots[att["data"]["beacon_block_root"]] += num_new_voters
            weights[slot] += num_new_voters

            voted_bitfields[slot][index] |= voters

        total_vote = sum(roots.values())
        reorged_roots = {r: v for r, v in roots.items() if r not in canonical_roots}
        reorgs[slot] = sum(reorged_roots.values())

    slots, y = dict_to_xy(reorgs)
    slots_weights, w = dict_to_xy(weights)
    assert np.all(slots == slots_weights)
    return slots, w, y


def plot_throughput(ax, slots, throughput):
    ax.bar(slots - MERGE_SLOT, throughput / 1000, width=1, color="red")
    ax.set_xlim(slots[0] - MERGE_SLOT, slots[-1] - MERGE_SLOT)

    ax.set_title("Attestations")

    ax.set_xlabel("Slot")
    ax.set_ylabel("Attestations [k/slot]")

    ax_top = ax.secondary_xaxis(
        "top", functions=(lambda s: s * 12 / 60 / 60, lambda t: t / 12 * 60 * 60)
    )
    ax_top.set_xlabel("Time [h]")

    ax_right = ax.secondary_yaxis(
        "right", functions=(lambda s: s / 12, lambda t: t * 12)
    )
    ax_right.set_ylabel("Attestations [k/s]")


def plot_throughput_histogram(ax, slots, throughput):
    mean = np.mean(throughput)
    outlier_mask = np.abs(throughput - mean) / mean < 0.05

    pre_merge_mask = slots < MERGE_SLOT
    post_merge_mask = slots >= MERGE_SLOT

    pre_masked = throughput[pre_merge_mask & outlier_mask]
    post_masked = throughput[post_merge_mask & outlier_mask]

    post_hist, bins = np.histogram(post_masked, bins=100)
    post_hist = post_hist / np.max(post_hist)
    pre_hist, _ = np.histogram(pre_masked, bins=bins)
    pre_hist = pre_hist / np.max(pre_hist)
    bin_width = bins[1] - bins[0]

    pre_mean = np.mean(pre_masked)
    pre_std = np.std(pre_masked)
    post_mean = np.mean(post_masked)
    post_std = np.std(post_masked)

    print(pre_std, post_std)

    ax_std = ax.twinx()
    ax_std.set_ylim(0, 1)
    ax_std.get_yaxis().set_ticks([])
    ax_std.errorbar(
        pre_mean,
        0.55,
        xerr=pre_std / 2,
        capsize=5,
        color="#16537e",
        marker="|",
    )
    ax_std.errorbar(
        post_mean, 0.45, xerr=post_std / 2, capsize=5, color="#bc5800", marker="|"
    )

    post_handle = ax.bar(
        bins[:-1],
        post_hist,
        align="edge",
        width=bin_width,
        color="C1",
    )
    pre_handle = ax.bar(
        bins[:-1],
        pre_hist,
        align="edge",
        width=bin_width,
        color="C0",
    )
    ax.step(
        bins[:-1],
        post_hist,
        color="C1",
        where="post",
    )

    ax.set_xlabel("Attestations per slot (k)")
    ax.set_ylabel("Frequency")
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: f"{x / 1000:.1f}")
    ax.xaxis.set_major_formatter(ticks)
    ax.legend([pre_handle, post_handle], ["Before merge", "After merge"])
    ax.set_title("Attestation inclusion rates")


def plot_delay_histogram(ax, delays):
    hist_dict_pre = collections.defaultdict(int)
    hist_dict_post = collections.defaultdict(int)
    for slot, ds in delays.items():
        for d in ds or []:
            count, delay = d
            if slot < MERGE_SLOT:
                hist_dict_pre[delay] += count
            else:
                hist_dict_post[delay] += count

    x_pre, y_pre = dict_to_xy(hist_dict_pre)
    x_post, y_post = dict_to_xy(hist_dict_post)

    print("immediate inclusions:")
    print("pre:", y_pre[0] / np.sum(y_pre))
    print("post:", y_post[0] / np.sum(y_post))
    print("average inclusion:")
    print("pre:", np.sum(y_pre * x_pre) / np.sum(y_pre))
    print("post:", np.sum(y_post * x_post) / np.sum(y_post))

    y_pre = y_pre[1:] / np.sum(y_pre[1:])
    y_post = y_post[1:] / np.sum(y_post[1:])
    x_pre = x_pre[1:]
    x_post = x_post[1:]

    handle_pre = ax.bar(x_pre - 0.2, y_pre, width=0.4, color="C0")
    handle_post = ax.bar(x_post + 0.2, y_post, width=0.4, color="C1")
    ax.set_xlim(1.5, 10.5)
    ax.set_xlabel("Inclusion delay [slots]")
    ax.set_ylabel("Frequency")
    ax.legend([handle_pre, handle_post], ["Before merge", "After merge"])
    ax.set_title("Attestation inclusion delays")


def plot_aggregation_steps(ax, agg_steps):
    num_steps_pre = []
    num_steps_post = []
    for (slot, _), d in agg_steps.items():
        for _, steps in d.items():
            if slot < MERGE_SLOT:
                num_steps_pre.append(len(steps))
            else:
                num_steps_post.append(len(steps))

    mean_pre = np.mean(num_steps_pre)
    mean_post = np.mean(num_steps_post)

    bins = np.arange(1, 8, 1)
    hist_pre, _ = np.histogram(num_steps_pre, bins=bins, density=True)
    hist_post, _ = np.histogram(num_steps_post, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    bar_width = bin_width / 2 * 0.8
    handle_pre = ax.bar(
        bins[:-1],
        hist_pre,
        align="edge",
        width=bar_width,
        color="C0",
    )
    handle_post = ax.bar(
        bins[:-1],
        hist_post,
        align="edge",
        width=-bar_width,
        color="C1",
    )
    ax.axvline(mean_pre, color=colors_darker[0])
    ax.axvline(mean_post, color=colors_darker[1])

    ax.legend([handle_pre, handle_post], ["Before merge", "After merge"])
    ax.set_xlabel("Number of attestations")
    ax.set_ylabel("Frequency")
    ax.set_title("Aggregation efficiency")


def plot_aggregation_effectiveness(ax, agg_steps):
    agg_effs_pre = []
    agg_effs_post = []
    for (slot, index), d in agg_steps.items():
        for key, steps in d.items():
            num_first_bits = np.sum(steps[0]["bits"])
            num_total_bits = np.sum(steps[-1]["cum_bits"])
            # eff = num_first_bits / num_total_bits
            eff = num_first_bits / len(steps[0]["bits"])
            if slot < MERGE_SLOT:
                agg_effs_pre.append(eff)
            else:
                agg_effs_post.append(eff)

    bins = np.linspace(0.95, 1.0, 6)
    hist_pre, _ = np.histogram(agg_effs_pre, bins=bins, density=True)
    hist_post, _ = np.histogram(agg_effs_post, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    bar_width = bin_width / 2
    handle_pre = ax.bar(
        bins[:-1] + bin_width / 2,
        hist_pre,
        align="edge",
        width=bar_width,
        color="C0",
    )
    handle_post = ax.bar(
        bins[:-1] + bin_width / 2,
        hist_post,
        align="edge",
        width=-bar_width,
        color="C1",
    )

    ax.legend([handle_pre, handle_post], ["Before merge", "After merge"])
    ax.set_xlabel("Fraction of attesters in heaviest attestation")
    ax.set_ylabel("Frequency")
    ax.set_title("Aggregation effectiveness")


def plot_aggregation_overlaps(ax, agg_steps):
    overlaps_pre = []
    overlaps_post = []
    for (slot, index), d in agg_steps.items():
        for key, steps in d.items():
            if len(steps) < 2:
                continue
            num_first_bits = np.sum(steps[0]["bits"])
            overlap = np.sum(steps[0]["bits"] & steps[1]["bits"])
            overlap_normed = overlap / len(steps[1]["bits"])
            if slot < MERGE_SLOT:
                overlaps_pre.append(overlap_normed)
            else:
                overlaps_post.append(overlap_normed)

    hist_pre, bins = np.histogram(overlaps_pre, bins=100, density=True)
    hist_post, _ = np.histogram(overlaps_post, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    handle_pre = ax.bar(bins[:-1], hist_pre, width=bin_width, align="edge", color="C0")
    handle_post = ax.bar(
        bins[:-1], hist_post, width=bin_width, align="edge", color="C1"
    )
    ax.step(bins[:-1], hist_pre, where="post", color="C0")
    # ax.step(bins[:-1], hist_post, where="post", color="C1")

    ax.legend([handle_pre, handle_post], ["Before merge", "After merge"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Frequency")
    ax.set_title("Aggregation overlaps")
    ax.set_ylim(0, max(np.max(hist_pre), np.max(hist_post)))


def plot_disagreements(ax, slots, weights, disagreements):
    x = slots - MERGE_SLOT
    assert np.sum(disagreements[1] == 0)
    assert np.sum(disagreements[2] == 0)
    y = disagreements[0]

    ax.bar(x, y * 100, width=1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 10)

    ax.set_xlabel("Slot")
    ax.set_ylabel("Disagreement [%]")
    ax.set_title("Disagreement")


def plot_disagreement_histogram(ax, slots, weights, disagreements):
    mask_pre = slots < MERGE_SLOT
    mask_post = ~mask_pre
    dis_pre = disagreements[0][mask_pre]
    dis_post = disagreements[0][mask_post]
    weights_pre = weights[mask_pre]
    weights_post = weights[mask_post]

    bins = np.linspace(0, 0.015, 101)
    hist_pre, _ = np.histogram(dis_pre, weights=weights_pre, bins=bins, density=True)
    hist_post, _ = np.histogram(dis_post, weights=weights_post, bins=bins, density=True)

    mean_pre = np.average(dis_pre, weights=weights_pre)
    mean_post = np.average(dis_post, weights=weights_post)
    median_pre = ws.numpy_weighted_median(dis_pre, weights=weights_pre)
    median_post = ws.numpy_weighted_median(dis_post, weights=weights_post)
    std_pre = np.sqrt(np.average((dis_pre - mean_pre) ** 2))
    std_post = np.sqrt(np.average((dis_post - mean_post) ** 2))

    x_scale = 100
    x = bins[:-1] * x_scale
    width = x[1] - x[0]
    handle_post = ax.bar(x, hist_post, width=width, align="edge", color="C1")
    handle_pre = ax.bar(x, hist_pre, width=width, align="edge", color="C0")
    ax.step(
        x,
        hist_post,
        color="C1",
        where="post",
    )

    print(mean_pre, median_pre, std_pre)
    print(mean_post, median_post, std_post)
    ax.axvline(median_pre * x_scale, color=colors_darker[0])
    ax.axvline(median_post * x_scale, color=colors_darker[1])

    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Disagreement [%]")
    ax.set_ylabel("Frequency")
    ax.set_title("Disagreement")
    ax.legend([handle_pre, handle_post], ["Before merge", "After merge"])


if __name__ == "__main__":
    print("loading...")
    # attestations = load_attestations(list(range(146870, 146880)))
    # attestations = load_attestations()
    # headers = load_headers()

    print("analyzing...")
    # slots, throughput = calc_throughput(attestations)
    # delays = calc_delays(attestations)
    # agg_steps = calc_aggregation_steps(attestations)
    # slots, weights, disagreements = calc_disagreements(attestations)
    # slots, weights, reorgs = calc_reorgs(attestations, headers)
    print(np.sum(reorgs) / np.sum(weights) * 100)

    print("plotting...")
    fig = plt.figure()
    # plot_throughput(fig.subplots(1, 1), slots, throughput)
    # plot_throughput_histogram(fig.subplots(1, 1), slots, throughput)
    # plot_delay_histogram(fig.subplots(1, 1), delays)
    # plot_aggregation_steps(fig.subplots(1, 1), agg_steps)
    # plot_aggregation_effectiveness(fig.subplots(1, 1), agg_steps)
    # plot_aggregation_overlaps(fig.subplots(1, 1), agg_steps)
    # plot_disagreements(fig.subplots(1, 1), slots, weights, disagreements)
    # plot_disagreement_histogram(fig.subplots(1, 1), slots, weights, disagreements)
    plt.tight_layout()

    print("exporting...")
    plt.savefig("plot.png", dpi=300)
    print("showing...")
    plt.show()
    print("done")
