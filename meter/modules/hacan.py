"""Core HACAN alignment operations described in the revised manuscript."""

import torch
import torch.nn.functional as F


def _caption_tokens(captions, cap_lens, index):
    """Return word tokens only (without BERT's [CLS]/[SEP] tokens)."""
    length = int(cap_lens[index])
    start = 1 if length > 2 else 0
    end = max(start + 1, length - 1) if length > 2 else max(1, length)
    return captions[index, start:end, :]


def _cosine_matrix(left, right, eps=1e-8):
    left = F.normalize(left, p=2, dim=-1, eps=eps)
    right = F.normalize(right, p=2, dim=-1, eps=eps)
    return torch.bmm(left, right.transpose(1, 2))


def dual_stream_completion_score(images, captions, cap_lens, eps=1e-8):
    """Local Relationship Guidance (LRG) from Eqs. (3)--(7)."""
    similarities = []
    num_images = images.size(0)

    for caption_index in range(captions.size(0)):
        tokens = _caption_tokens(captions, cap_lens, caption_index)
        tokens = tokens.unsqueeze(0).expand(num_images, -1, -1)

        visual_context = images.mean(dim=1)
        text_gate = torch.sigmoid(
            torch.sum(tokens * visual_context.unsqueeze(1), dim=-1, keepdim=True)
        )
        contextual_tokens = tokens * text_gate

        anchor_scores = _cosine_matrix(images, contextual_tokens, eps=eps)
        anchor_index = anchor_scores.argmax(dim=-1)
        anchors = torch.gather(
            contextual_tokens,
            1,
            anchor_index.unsqueeze(-1).expand(-1, -1, contextual_tokens.size(-1)),
        )
        visual_gate = torch.sigmoid(
            torch.sum(images * anchors, dim=-1, keepdim=True)
        )
        contextual_images = images * visual_gate

        pair_scores = _cosine_matrix(contextual_images, contextual_tokens, eps=eps)
        attention = F.softmax(pair_scores, dim=1)
        score = (attention * pair_scores).sum(dim=1).mean(dim=1)
        similarities.append(score.unsqueeze(1))

    return torch.cat(similarities, dim=1)


def _key_semantic_filter_one_way(query, candidates, eps=1e-8):
    raw_scores = _cosine_matrix(query, candidates, eps=eps)
    query_mean = raw_scores.mean(dim=2, keepdim=True)
    candidate_mean = raw_scores.mean(dim=1, keepdim=True)
    deviation = ((raw_scores - query_mean) + (raw_scores - candidate_mean)) / 2

    candidate_residual = candidates - candidates.mean(dim=1, keepdim=True)
    residual_scores = _cosine_matrix(query, candidate_residual, eps=eps)
    weighted_scores = residual_scores * torch.sign(deviation)
    weighted_scores = torch.where(
        deviation.abs() > eps, weighted_scores, torch.zeros_like(weighted_scores)
    )
    return weighted_scores.max(dim=2).values.mean(dim=1)


def key_semantic_filter_score(images, captions, cap_lens, direction="i2t"):
    """Key Semantic Filter (KSF) from Eqs. (8)--(12)."""
    if direction not in {"i2t", "t2i"}:
        raise ValueError("direction must be 'i2t' or 't2i'")

    similarities = []
    num_images = images.size(0)
    for caption_index in range(captions.size(0)):
        tokens = _caption_tokens(captions, cap_lens, caption_index)
        tokens = tokens.unsqueeze(0).expand(num_images, -1, -1)
        if direction == "i2t":
            score = _key_semantic_filter_one_way(images, tokens)
        else:
            score = _key_semantic_filter_one_way(tokens, images)
        similarities.append(score.unsqueeze(1))
    return torch.cat(similarities, dim=1)


def hacan_similarity(images_h, captions_h, images_m, captions_m, cap_lens, direction):
    """Fine-grained score: KSF plus the two cross-layer LRG streams."""
    return (
        key_semantic_filter_score(images_h, captions_h, cap_lens, direction)
        + dual_stream_completion_score(images_m, captions_h, cap_lens)
        + dual_stream_completion_score(images_h, captions_m, cap_lens)
    )


def global_contrastive_divergence(
    images, captions, text_negative_index, image_negative_index, margin=0.2
):
    """Global Contrastive Divergence loss from Eqs. (14)--(17)."""
    positive = F.cosine_similarity(images, captions, dim=-1)
    hard_text = captions[text_negative_index]
    hard_image = images[image_negative_index]

    image_to_text = (margin + F.cosine_similarity(images, hard_text, dim=-1) - positive).clamp(min=0)
    text_to_image = (margin + F.cosine_similarity(hard_image, captions, dim=-1) - positive).clamp(min=0)
    triplet = image_to_text.mean() + text_to_image.mean()

    image_negative_similarity = F.cosine_similarity(images, hard_image, dim=-1)
    paired_text_similarity = F.cosine_similarity(
        captions, captions[image_negative_index], dim=-1
    )
    intra_image = (
        (margin + image_negative_similarity - positive).clamp(min=0)
        + (image_negative_similarity - paired_text_similarity).abs()
    ).mean()

    text_negative_similarity = F.cosine_similarity(captions, hard_text, dim=-1)
    paired_image_similarity = F.cosine_similarity(
        images, images[text_negative_index], dim=-1
    )
    intra_text = (
        (margin + text_negative_similarity - positive).clamp(min=0)
        + (paired_image_similarity - text_negative_similarity).abs()
    ).mean()

    unpaired_mask = text_negative_index != image_negative_index
    if unpaired_mask.any():
        unpaired_similarity = F.cosine_similarity(
            hard_image[unpaired_mask], hard_text[unpaired_mask], dim=-1
        )
        unpaired = (
            margin + unpaired_similarity - positive[unpaired_mask]
        ).clamp(min=0).mean()
    else:
        unpaired = positive.new_zeros(())

    return triplet + intra_image + intra_text + unpaired
