from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PERTURBATION_TIERS = (
    "clean",
    "typo_light",
    "typo_heavy",
    "grammar_light",
    "mixed_light",
    "mixed_heavy",
    "keyword_synonym",
    "keyword_neutralize",
    "keyword_swap_type",
    "identifier_mask",
    "structure_preserve_lexical_change",
)

LEXICAL_PROBE_TIERS = (
    "keyword_synonym",
    "keyword_neutralize",
    "keyword_swap_type",
    "identifier_mask",
    "structure_preserve_lexical_change",
)

_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_IDENTIFIER_STOPWORDS = _FUNCTION_WORDS | {
    "function",
    "python",
    "code",
    "return",
    "write",
    "find",
    "given",
    "using",
    "implement",
    "compute",
    "calculate",
    "determine",
}
_WORD_RE = re.compile(r"[A-Za-z]+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
_TEXTATTACK_AUGMENTER_CACHE: dict[tuple[str, float, int], Any] = {}


@dataclass(frozen=True)
class PerturbationSpec:
    augmenter_kind: str
    pct_words_to_swap: float
    transformations_per_example: int
    pre_normalize_grammar: bool = False


_TIER_SPECS: dict[str, PerturbationSpec] = {
    "typo_light": PerturbationSpec("charswap", pct_words_to_swap=0.10, transformations_per_example=1),
    "typo_heavy": PerturbationSpec("charswap", pct_words_to_swap=0.30, transformations_per_example=2),
    "grammar_light": PerturbationSpec("deletion", pct_words_to_swap=0.18, transformations_per_example=1, pre_normalize_grammar=True),
    "mixed_light": PerturbationSpec("mixed", pct_words_to_swap=0.12, transformations_per_example=1, pre_normalize_grammar=True),
    "mixed_heavy": PerturbationSpec("mixed", pct_words_to_swap=0.28, transformations_per_example=2, pre_normalize_grammar=True),
}


def perturb_queries(
    queries: list[str],
    tier: str,
    seed: int,
    lexical_map_path: str | None = None,
) -> list[str]:
    normalized_tier = _normalize_tier(tier)
    if normalized_tier == "clean":
        return list(queries)
    if normalized_tier in LEXICAL_PROBE_TIERS:
        lexical_map = load_lexical_map(lexical_map_path)
        return [
            _perturb_with_keyword_probe(query, tier=normalized_tier, seed=seed, query_index=index, lexical_map=lexical_map)
            for index, query in enumerate(queries)
        ]

    spec = _TIER_SPECS[normalized_tier]
    prepared = [_grammar_pre_normalize(query) if spec.pre_normalize_grammar else query for query in queries]

    try:
        augmenter = _get_textattack_augmenter(spec)
    except ModuleNotFoundError:
        augmenter = None
    except Exception:
        augmenter = None

    if augmenter is None:
        return [
            _perturb_with_manual_fallback(text, tier=normalized_tier, seed=seed, query_index=index)
            for index, text in enumerate(prepared)
        ]

    perturbed: list[str] = []
    try:
        for index, text in enumerate(prepared):
            perturbed.append(
                _perturb_with_textattack(
                    text,
                    spec=spec,
                    seed=seed,
                    query_index=index,
                    augmenter=augmenter,
                )
            )
        return perturbed
    except Exception:
        return [
            _perturb_with_manual_fallback(text, tier=normalized_tier, seed=seed, query_index=index)
            for index, text in enumerate(prepared)
        ]


def perturb_query(
    query: str,
    tier: str,
    seed: int,
    query_index: int = 0,
    lexical_map_path: str | None = None,
) -> str:
    normalized_tier = _normalize_tier(tier)
    if normalized_tier == "clean":
        return query
    if normalized_tier in LEXICAL_PROBE_TIERS:
        return _perturb_with_keyword_probe(
            query,
            tier=normalized_tier,
            seed=seed,
            query_index=query_index,
            lexical_map=load_lexical_map(lexical_map_path),
        )

    spec = _TIER_SPECS[normalized_tier]
    text = _grammar_pre_normalize(query) if spec.pre_normalize_grammar else query
    try:
        return _perturb_with_textattack(
            text,
            spec=spec,
            seed=seed,
            query_index=query_index,
            augmenter=_get_textattack_augmenter(spec),
        )
    except ModuleNotFoundError:
        return _perturb_with_manual_fallback(text, tier=normalized_tier, seed=seed, query_index=query_index)
    except Exception:
        return _perturb_with_manual_fallback(text, tier=normalized_tier, seed=seed, query_index=query_index)


def load_lexical_map(path: str | None = None) -> dict[str, dict[str, Any]]:
    return _load_lexical_map_cached(str(Path(path).expanduser().resolve()) if path else "")


@lru_cache(maxsize=4)
def _load_lexical_map_cached(path_key: str) -> dict[str, dict[str, Any]]:
    lexical_path = (
        Path(path_key)
        if path_key
        else Path(__file__).resolve().parent / "lexical_replacements.json"
    )
    payload = json.loads(lexical_path.read_text(encoding="utf-8"))
    return {str(key).lower(): dict(value) for key, value in payload.items() if isinstance(value, dict)}


def _normalize_tier(tier: str) -> str:
    normalized = tier.strip().lower()
    if normalized not in PERTURBATION_TIERS:
        raise ValueError(f"Unsupported perturbation tier: {tier}")
    return normalized


def _get_textattack_augmenter(spec: PerturbationSpec) -> Any:
    cache_key = (spec.augmenter_kind, spec.pct_words_to_swap, spec.transformations_per_example)
    cached = _TEXTATTACK_AUGMENTER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    augmenter = _build_textattack_augmenter(spec)
    _TEXTATTACK_AUGMENTER_CACHE[cache_key] = augmenter
    return augmenter


def _build_textattack_augmenter(spec: PerturbationSpec) -> Any:
    if spec.augmenter_kind == "charswap":
        from textattack.augmentation.recipes import CharSwapAugmenter

        return CharSwapAugmenter(
            pct_words_to_swap=spec.pct_words_to_swap,
            transformations_per_example=spec.transformations_per_example,
            high_yield=False,
            fast_augment=True,
        )
    if spec.augmenter_kind == "deletion":
        from textattack.augmentation.recipes import DeletionAugmenter

        return DeletionAugmenter(
            pct_words_to_swap=spec.pct_words_to_swap,
            transformations_per_example=spec.transformations_per_example,
            high_yield=False,
            fast_augment=True,
        )
    if spec.augmenter_kind == "mixed":
        from textattack.augmentation import Augmenter
        from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapNeighboringCharacterSwap,
            WordSwapQWERTY,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
            WordSwapRandomCharacterSubstitution,
        )

        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion(),
                WordSwapRandomCharacterSubstitution(),
                WordSwapQWERTY(),
            ]
        )
        return Augmenter(
            transformation=transformation,
            constraints=[RepeatModification(), StopwordModification()],
            pct_words_to_swap=spec.pct_words_to_swap,
            transformations_per_example=spec.transformations_per_example,
            high_yield=False,
            fast_augment=True,
        )
    raise ValueError(f"Unsupported TextAttack augmenter kind: {spec.augmenter_kind}")


def _perturb_with_textattack(text: str, *, spec: PerturbationSpec, seed: int, query_index: int, augmenter: Any) -> str:
    random.seed(f"{seed}:{query_index}:{spec.augmenter_kind}")
    augmented = augmenter.augment(text)
    if isinstance(augmented, tuple):
        augmented = augmented[0]
    if not augmented:
        return text
    return _SPACE_RE.sub(" ", str(augmented[0])).strip()


def _perturb_with_keyword_probe(
    text: str,
    *,
    tier: str,
    seed: int,
    query_index: int,
    lexical_map: dict[str, dict[str, Any]],
) -> str:
    rng = random.Random(f"{seed}:{query_index}:{tier}")
    words = list(_WORD_RE.finditer(text))
    if not words:
        return text

    replacements: list[tuple[re.Match[str], str]] = []
    used_terms: set[str] = set()
    for match in words:
        token = match.group(0)
        replacement = _keyword_replacement_for_token(token, tier=tier, rng=rng, lexical_map=lexical_map)
        if replacement is None or replacement.lower() == token.lower():
            continue
        key = token.lower()
        if key in used_terms:
            continue
        used_terms.add(key)
        replacements.append((match, replacement))

    if not replacements and tier == "identifier_mask":
        return _identifier_mask_fallback(text, rng=rng, lexical_map=lexical_map)
    if not replacements:
        return text

    pieces: list[str] = []
    last_end = 0
    for match, replacement in replacements:
        pieces.append(text[last_end:match.start()])
        pieces.append(_match_case(match.group(0), replacement))
        last_end = match.end()
    pieces.append(text[last_end:])
    return _SPACE_RE.sub(" ", "".join(pieces)).strip()


def _keyword_replacement_for_token(
    token: str,
    *,
    tier: str,
    rng: random.Random,
    lexical_map: dict[str, dict[str, Any]],
) -> str | None:
    normalized = token.lower()
    entry = lexical_map.get(normalized)

    if tier == "identifier_mask":
        if normalized in lexical_map:
            neutral = _choose(entry, "neutralize", rng) if entry else None
            return neutral or "value"
        if normalized not in _IDENTIFIER_STOPWORDS and len(normalized) >= 4:
            return rng.choice(("item", "value", "result", "data"))
        return None

    if entry is None:
        if tier in {"keyword_synonym", "structure_preserve_lexical_change"}:
            return _wordnet_synonym(normalized)
        return None

    if tier == "keyword_synonym":
        return _choose(entry, "synonyms", rng) or _wordnet_synonym(normalized)
    if tier == "keyword_neutralize":
        return _choose(entry, "neutralize", rng)
    if tier == "keyword_swap_type":
        return _choose(entry, "swap_type", rng)
    if tier == "structure_preserve_lexical_change":
        return _choose(entry, "synonyms", rng) or _choose(entry, "neutralize", rng) or _wordnet_synonym(normalized)
    return None


def _choose(entry: dict[str, Any], field: str, rng: random.Random) -> str | None:
    values = entry.get(field)
    if not isinstance(values, list) or not values:
        return None
    return str(rng.choice(values))


def _wordnet_synonym(token: str) -> str | None:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None

    candidates: list[str] = []
    for synset in wn.synsets(token):
        for lemma in synset.lemma_names():
            candidate = lemma.replace("_", " ").strip().lower()
            if " " in candidate:
                continue
            if candidate == token or not candidate.isalpha():
                continue
            candidates.append(candidate)
    return sorted(set(candidates))[0] if candidates else None


def _identifier_mask_fallback(text: str, *, rng: random.Random, lexical_map: dict[str, dict[str, Any]]) -> str:
    words = list(_WORD_RE.finditer(text))
    target_match = None
    for match in words:
        token = match.group(0).lower()
        if token in _IDENTIFIER_STOPWORDS:
            continue
        if len(token) < 4:
            continue
        target_match = match
        break
    if target_match is None:
        return text
    replacement = _keyword_replacement_for_token(target_match.group(0), tier="identifier_mask", rng=rng, lexical_map=lexical_map)
    if replacement is None:
        return text
    return (
        text[: target_match.start()]
        + _match_case(target_match.group(0), replacement)
        + text[target_match.end() :]
    )


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _grammar_pre_normalize(text: str) -> str:
    lowered = text.lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    tokens = [token for token in lowered.split() if token not in _FUNCTION_WORDS]
    if not tokens:
        tokens = lowered.split() or [text]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def _perturb_with_manual_fallback(text: str, *, tier: str, seed: int, query_index: int) -> str:
    rng = random.Random(f"{seed}:{query_index}:{tier}")
    if tier == "typo_light":
        return _apply_typo_noise(text, rng=rng, token_ratio=0.10, edits_per_token=(1, 1))
    if tier == "typo_heavy":
        return _apply_typo_noise(text, rng=rng, token_ratio=0.30, edits_per_token=(1, 2))
    if tier == "grammar_light":
        return _SPACE_RE.sub(" ", text).strip()
    if tier == "mixed_light":
        return _apply_typo_noise(text, rng=rng, token_ratio=0.12, edits_per_token=(1, 1))
    if tier == "mixed_heavy":
        return _apply_typo_noise(text, rng=rng, token_ratio=0.28, edits_per_token=(1, 2))
    return text


def _apply_typo_noise(
    text: str,
    *,
    rng: random.Random,
    token_ratio: float,
    edits_per_token: tuple[int, int],
) -> str:
    words = list(_WORD_RE.finditer(text))
    if not words:
        return text

    candidate_indices = [index for index, match in enumerate(words) if len(match.group(0)) >= 3]
    if not candidate_indices:
        return text

    n_targets = max(1, round(len(candidate_indices) * token_ratio))
    target_indices = set(rng.sample(candidate_indices, k=min(n_targets, len(candidate_indices))))

    pieces: list[str] = []
    last_end = 0
    for index, match in enumerate(words):
        pieces.append(text[last_end:match.start()])
        token = match.group(0)
        if index in target_indices:
            n_edits = rng.randint(edits_per_token[0], edits_per_token[1])
            pieces.append(_edit_token(token, rng=rng, n_edits=n_edits))
        else:
            pieces.append(token)
        last_end = match.end()
    pieces.append(text[last_end:])
    return _SPACE_RE.sub(" ", "".join(pieces)).strip()


def _edit_token(token: str, *, rng: random.Random, n_edits: int) -> str:
    mutated = token
    for _ in range(max(1, n_edits)):
        mutated = _single_edit(mutated, rng=rng)
    return mutated


def _single_edit(token: str, *, rng: random.Random) -> str:
    if len(token) < 2:
        return token

    edit_type = rng.choice(("swap", "delete", "insert", "replace"))
    letters = string.ascii_lowercase
    pos = rng.randrange(len(token))

    if edit_type == "swap" and len(token) >= 2:
        if pos == len(token) - 1:
            pos -= 1
        chars = list(token)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)
    if edit_type == "delete" and len(token) > 3:
        return token[:pos] + token[pos + 1 :]
    if edit_type == "insert":
        char = rng.choice(letters)
        return token[:pos] + char + token[pos:]
    char = rng.choice(letters)
    return token[:pos] + char + token[pos + 1 :]
