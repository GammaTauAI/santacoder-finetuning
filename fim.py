from pathlib import Path
from tree_sitter import Language, Parser, Node
import functools
import random

import numpy as np
from numpy.random import RandomState

from typing import List, Tuple, Any, Optional

Language.build_library(
    f"{Path(__file__).parent}/build/languages.so",
    [f"{Path(__file__).parent}/tree-sitter-typescript/typescript"]
)
TS_LANGUAGE = Language(
    f"{Path(__file__).parent}/build/languages.so", 'typescript')
PARSER = Parser()
PARSER.set_language(TS_LANGUAGE)

# this is expensive so we cache it


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        _, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map[
            "additional_special_tokens"
        ]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


def build(content):
    def create_query(query):
        return TS_LANGUAGE.query(query)

    def str_to_tree(contents):
        return PARSER.parse(bytes(contents, "utf-8"))

    def is_child_type_annotation(node):
        """Checks if any of the parent nodes is an annotation node."""
        node = node.parent
        while node is not None:
            if node.type == "type_annotation" or node.type == "opting_type_annotation" or node.type == "omitting_type_annotation":
                return True
            node = node.parent
        return False

    QUERY = create_query("""
[
  (type_annotation) @annotation
  (opting_type_annotation) @annotation
  (omitting_type_annotation) @annotation
]
""")
    tree = str_to_tree(content)

    # Each capture has a start_byte and end_byte; these are the indices of the
    # type annotation. We want to invert these indices, i.e. get the substrings
    # between the captures (and also the substring before the first capture and
    # the substring after the last capture).
    captures: List[Node] = QUERY.captures(tree.root_node)

    # Need to operate on byte string, not characters
    content_bytes = content.encode("utf-8")

    # Flatten the capture indices into a list (but skip over child type
    # annotations). But we also want to prepend 0 and append the last index of
    # content, so we can re-pair the indices,
    # e.g. [(s1, e1), (s2, e2)]
    #   -> [0, s1, e1, s2, e2, n]
    #   -> [(0, s1), (e1, s2), (e2, n)]
    indices = [0] + [i
                     for c in captures
                     for i in [c[0].start_byte, c[0].end_byte]
                     if not is_child_type_annotation(c[0])]
    indices.append(len(content_bytes))

    # We zip the list with itself (offset by 1), moving by 2 elements each time.
    chunks = []
    for s, e in zip(indices[::2], indices[1::2]):
        chunks.append(content_bytes[s:e].decode("utf-8"))
    new_content = "".join(chunks)

    return new_content


def get_prefix_middle_suffix(np_rng: RandomState, sample: bytes) -> Optional[Tuple[Tuple[bytes, bytes, bytes], RandomState]]:
    def is_child_type_annotation(node):
        """Checks if any of the parent nodes is an annotation node."""
        node = node.parent
        while node is not None:
            if node.type == "type_annotation" or node.type == "opting_type_annotation" or node.type == "omitting_type_annotation":
                return True
            node = node.parent
        return False

    QUERY = TS_LANGUAGE.query("""
[
  (type_annotation) @annotation
  (opting_type_annotation) @annotation
  (omitting_type_annotation) @annotation
]
""")
    tree = PARSER.parse(sample)

    # Each capture has a start_byte and end_byte; these are the indices of the
    # type annotation. We want to invert these indices, i.e. get the substrings
    # between the captures (and also the substring before the first capture and
    # the substring after the last capture).
    captures: List[Node] = QUERY.captures(tree.root_node)

    captures_no_child: List[int] = []
    for i, (node, _) in enumerate(captures):
        if not is_child_type_annotation(node):
            captures_no_child += [i]

    if len(captures_no_child) == 0:
        return None
    random_pick_i = np_rng.choice(captures_no_child)

    prefix_str: bytes = sample[:captures[random_pick_i][0].start_byte]
    middle_str: bytes = sample[captures[random_pick_i]
                               [0].start_byte:captures[random_pick_i][0].end_byte]
    if middle_str.startswith(b": "):
        prefix_str += b": "
        middle_str = middle_str[2:]
    suffix_str: bytes = b""
    l = len(captures)
    for i in range(random_pick_i, l - 1):
        suffix_str += sample[captures[i]
                             [0].end_byte:captures[i + 1][0].start_byte]
    suffix_str += sample[captures[l - 1][0].end_byte:]

    return (prefix_str, middle_str, suffix_str), np_rng


# Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    tokenizer,
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    if np_rng.binomial(1, fim_rate):
        decoded_bytes: bytes = tokenizer.decode(sample)
        assert isinstance(decoded_bytes, bytes)  # just making sure

        res = get_prefix_middle_suffix(np_rng, decoded_bytes)
        if res is None:
            return None, np_rng

        (prefix_str, middle_str, suffix_str), np_rng = res

        prefix = np.array(tokenizer.encode(prefix_str))
        middle = np.array(tokenizer.encode(middle_str))
        suffix = np.array(tokenizer.encode(suffix_str))

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample

    return list(new_sample), np_rng


if __name__ == "__main__":  # some unit tests
    import os
    rng = np.random.RandomState(seed=int(os.urandom(4).hex(), 16))
    sample = """
    function foo(x: number, y: number): number {
        return x + y;
    }

    // some unicode to mess things up
    // 😀 😃 😄 😁 😆 😅

    function foo2(x: number, y: number): number {
        return x + y;
    }
    """
    bytes_sample = bytes(sample, "utf-8")
    print("sample:", sample)
    print("bytes_sample:", bytes_sample)

    print("get_prefix_middle_suffix:")
    res = get_prefix_middle_suffix(rng, bytes_sample)
    if res is not None:
        (prefix_str, middle_str, suffix_str), rng = res
        print("prefix_str:", prefix_str.decode("utf-8"))
        print("middle_str:", middle_str.decode("utf-8"))
        print("suffix_str:", suffix_str.decode("utf-8"))
