"""Passage annotation: which chunks of a filing actually contain an answer.

This package sits between the two sides of the benchmark without joining them.
It is handed truth values as plain floats and filing chunks as plain text, and
it emits chunk ids. It never reads the database, and nothing under ``agents/``
may import it: the annotations are an answer key for scoring retrieval, and an
agent that could read them would be scored on a lookup.
"""
