#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-05 09:16:34
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import argparse


def add_io_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:  # pragma: no cover
    """
    Add input/output arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="Whether to run in debug mode",
        default=False,
    )
    return parser


def add_meta_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:  # pragma: no cover
    """
    Add meta arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        help="""Batch size to use for dataset iteration. Mainly for memory efficiency.""",
        default=10000,
    ),
    return parser


def add_minhash_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:  # pragma: no cover
    """
    Add MinHash arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser with added arguments.
    """
    parser.add_argument(
        "--ngram",
        type=int,
        default=5,
        help="Ngram size to use in MinHash.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum number of tokens to use in MinHash. Shorter documents will be filtered out.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed to use in MinHash")
    parser.add_argument(
        "--num_perm",
        type=int,
        default=512,
        help="Number of permutations to use in MinHash",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Jaccard similarity threshold to use in MinHashLSH",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=None,
        help="Number of bands",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
        help="Number of rows per band",
    )

    return parser
