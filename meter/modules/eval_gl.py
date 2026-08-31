"""Hierarchical Inference Strategy (HIS) for HACAN retrieval."""

import numpy as np


def _recall_metrics(ranks):
    r1 = 100.0 * np.count_nonzero(ranks < 1) / len(ranks)
    r5 = 100.0 * np.count_nonzero(ranks < 5) / len(ranks)
    r10 = 100.0 * np.count_nonzero(ranks < 10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1, r5, r10, medr, meanr, 0.0, 0.0


def i2t_gl(
    full_img_emb_aggrs,
    full_cap_emb_aggrs,
    img_embs,
    cap_embs,
    img_emb_fusions,
    cap_emb_fusions,
    img_lengths,
    cap_lengths,
    npts=None,
    return_ranks=True,
    ndcg_scorer=None,
    fold_index=0,
    measure="dot",
    sim_function=None,
    sim_function_new=None,
    cap_batches=1,
    pl_module=None,
    fold5=-1,
    topk=50,
    weight=None,
):
    """Image-to-text: global Top-K recall followed by HACAN re-ranking."""
    if npts is None:
        npts = img_embs.shape[0] // 5

    images_h = np.asarray(img_embs)[::5]
    images_m = np.asarray(img_emb_fusions)[::5]
    image_global = np.asarray(full_img_emb_aggrs)[::5]
    coarse_scores = np.matmul(image_global, np.asarray(full_cap_emb_aggrs).T)
    topk = min(topk, cap_embs.shape[0])

    ranks = np.full(npts, np.inf)
    top1 = np.full(npts, -1, dtype=np.int64)
    for index in range(npts):
        candidates = np.argsort(coarse_scores[index])[::-1][:topk]
        image_h = images_h[index : index + 1]
        image_m = images_m[index : index + 1]
        candidate_lengths = np.asarray(cap_lengths)[candidates]

        ksf = sim_function_new(
            image_h, np.asarray(cap_embs)[candidates], candidate_lengths, pl_module
        )
        lrg_mh = sim_function(
            image_m, np.asarray(cap_embs)[candidates], candidate_lengths, pl_module
        )
        lrg_hm = sim_function(
            image_h,
            np.asarray(cap_emb_fusions)[candidates],
            candidate_lengths,
            pl_module,
        )
        fine_scores = (ksf + lrg_mh + lrg_hm).reshape(-1)
        ranking = candidates[np.argsort(fine_scores)[::-1]]
        top1[index] = ranking[0]

        ground_truth = np.arange(5 * index, 5 * index + 5)
        positions = np.flatnonzero(np.isin(ranking, ground_truth))
        if positions.size:
            ranks[index] = positions.min()

    metrics = _recall_metrics(ranks)
    if return_ranks:
        return metrics, (ranks, top1), coarse_scores
    return metrics, coarse_scores


def t2i_gl(
    full_img_emb_aggrs,
    full_cap_emb_aggrs,
    img_embs,
    cap_embs,
    img_emb_fusions,
    cap_emb_fusions,
    img_lengths,
    cap_lengths,
    npts=None,
    return_ranks=True,
    ndcg_scorer=None,
    fold_index=0,
    measure="dot",
    sim_function=None,
    sim_function_new=None,
    cap_batches=1,
    pl_module=None,
    sims=None,
    topk=50,
    weight=None,
):
    """Text-to-image: global Top-K recall followed by HACAN re-ranking."""
    if npts is None:
        npts = img_embs.shape[0] // 5

    images_h = np.asarray(img_embs)[::5]
    images_m = np.asarray(img_emb_fusions)[::5]
    image_global = np.asarray(full_img_emb_aggrs)[::5]
    coarse_scores = np.matmul(image_global, np.asarray(full_cap_emb_aggrs).T).T
    topk = min(topk, images_h.shape[0])

    ranks = np.full(5 * npts, np.inf)
    top5 = np.full((5 * npts, 5), -1, dtype=np.int64)
    for caption_index in range(5 * npts):
        candidates = np.argsort(coarse_scores[caption_index])[::-1][:topk]
        caption_h = np.asarray(cap_embs)[caption_index : caption_index + 1]
        caption_m = np.asarray(cap_emb_fusions)[caption_index : caption_index + 1]
        caption_length = [cap_lengths[caption_index]]

        ksf = sim_function_new(
            images_h[candidates], caption_h, caption_length, pl_module
        )
        lrg_mh = sim_function(
            images_m[candidates], caption_h, caption_length, pl_module
        )
        lrg_hm = sim_function(
            images_h[candidates], caption_m, caption_length, pl_module
        )
        fine_scores = (ksf + lrg_mh + lrg_hm).reshape(-1)
        ranking = candidates[np.argsort(fine_scores)[::-1]]
        top5[caption_index, : min(5, len(ranking))] = ranking[:5]

        correct_image = caption_index // 5
        position = np.flatnonzero(ranking == correct_image)
        if position.size:
            ranks[caption_index] = position[0]

    metrics = _recall_metrics(ranks)
    if return_ranks:
        return metrics, (ranks, top5)
    return metrics
