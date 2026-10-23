"""A6.4 gold-free output normalization; original rule preserved verbatim."""
import re
LEADING_BOOLEAN = re.compile(r'^\s*(yes|no)([,.:;!?]+)(?=\s|$)', re.I)

def surface_repair(answer):
    """Minimal surface change: make a leading Yes/No an isolated whitespace token.

    Preserve the rest of the response, including any later self-correction.
    Do not infer a new Yes/No label or use question labels.
    """
    s = answer.strip()
    m = LEADING_BOOLEAN.match(s)
    if not m:
        return s, 'unchanged'
    first = m.group(1).capitalize()
    tail = s[m.end():].lstrip()
    return first + ('\n' + tail if tail else ''), 'leading_boolean_punctuation'


