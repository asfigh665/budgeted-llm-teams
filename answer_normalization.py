"""Candidate normalization and task answer grading."""
from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fractions import Fraction
from pathlib import Path
from hashlib import sha256
import numpy as np
_WS = re.compile('\\s+')
_DROP_CMD = re.compile('\\\\(left|right)\\s*')
_MMA_OPT = re.compile('(?:answer\\s*(?:is|:)\\s*|option\\s*|^|\\()\\s*[\\(\\[]?([A-Ja-j])[\\)\\].:]?\\s*$', re.I)
_BOXED = re.compile('\\\\boxed\\s*\\{([^{}]*)\\}')
_ASSIGN = re.compile('^(?:[A-Za-z]|\\\\?[A-Za-z]_\\d+)=(.+)$')
_NUM = re.compile('^[+-]?\\d+(\\.\\d+)?$')
_FRAC = re.compile('^([+-]?)\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}$')
_SLASH = re.compile('^([+-]?)\\(?([+-]?\\d+)\\)?/\\(?([+-]?\\d+)\\)?$')
_DECFRAC = re.compile('^([+-]?)(\\d+(?:\\.\\d+)?)/(\\d+(?:\\.\\d+)?)$')
def _fold(s: str) -> str:
    s = s.replace('\\\\', '\\')
    s = _DROP_CMD.sub('', s)
    s = s.replace('$', '')
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = s.replace('\\leq', '\\le').replace('\\geq', '\\ge')
    s = s.replace('−', '-')
    s = s.replace('π', '\\pi')
    s = _WS.sub('', s)
    s = s.rstrip('.')
    while s.startswith('{') and s.endswith('}') and (s.count('{') == 1):
        s = s[1:-1]
    return s

def normalize_answer(x: Any, kind: str='generic') -> str:
    if x is None:
        return ''
    s = str(x).strip()
    if kind == 'mmlu_pro':
        return _extract_option(s)
    return _fold(s)

def _extract_option(s: str) -> str:
    if not s:
        return ''
    low = s.lower()
    if any((w in low for w in ('not sure', 'cannot determine', 'unknown', 'no idea', 'unable to', 'insufficient'))):
        return ''
    b = _BOXED.findall(s)
    if b:
        cand = _fold(b[-1]).upper()
        if len(cand) == 1 and cand in 'ABCDEFGHIJ':
            return cand
    patterns = ('answer\\s*(?:is|:)\\s*[\\(\\[]?([a-j])\\b', 'option\\s*[\\(\\[]?([a-j])\\b', '^\\s*[\\(\\[]([a-j])[\\)\\]]', '^\\s*([a-j])[\\)\\].:]')
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return m.group(1).upper()
    core = _fold(s).upper()
    if len(core) == 1 and core in 'ABCDEFGHIJ':
        return core
    m = re.search('(?:^|\\s)([a-j])\\s*[.!]?\\s*$', low)
    if m:
        return m.group(1).upper()
    return ''

def answers_equal(pred: Any, gold: Any, kind: str='generic') -> bool:
    if pred is None or gold is None:
        return False
    a_raw, b_raw = (str(pred).strip(), str(gold).strip())
    if not a_raw or not b_raw:
        return False
    a = normalize_answer(a_raw, kind)
    b = normalize_answer(b_raw, kind)
    if not a or not b:
        return False
    if _cmp(a, b):
        return True
    av, bv = (_assignment_value(a), _assignment_value(b))
    if av is not None and bv is None:
        return _cmp(av, b)
    if bv is not None and av is None:
        return _cmp(a, bv)
    if av is not None and bv is not None:
        return _cmp(av, bv)
    return False

def _assignment_value(s: str) -> Optional[str]:
    m = _ASSIGN.match(s)
    return m.group(1) if m else None

def _cmp(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    fa, fb = (_as_fraction(a), _as_fraction(b))
    if fa is not None and fb is not None:
        return fa == fb
    return False

def _as_fraction(s: str) -> Optional[Fraction]:
    if not s:
        return None
    m = _FRAC.match(s)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        try:
            num = Fraction(m.group(2))
            den = Fraction(m.group(3))
            return sign * num / den if den else None
        except (ValueError, ZeroDivisionError):
            return None
    for pat in (_SLASH, _DECFRAC):
        m = pat.match(s)
        if m:
            sign = -1 if m.group(1) == '-' else 1
            try:
                num = Fraction(m.group(2))
                den = Fraction(m.group(3))
                return sign * num / den if den else None
            except (ValueError, ZeroDivisionError):
                return None
    if _NUM.match(s):
        try:
            return Fraction(s)
        except ValueError:
            return None
    return None
