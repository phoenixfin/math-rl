"""
formula.py — Symbolic formula algebra for propositional logic.

Grammar:
    formula = var | ¬formula | (formula→formula)
    var     = p | q | r

All compound formulas are fully parenthesized.
Negation of atomic: ¬p  (no extra parens)
Negation of compound: ¬(p→q)
Implication: (A→B)
"""

from typing import Optional, Tuple, List, Dict, Set


VARS = ['p', 'q', 'r']

# ── Constructors ──────────────────────────────────────────────────────────────

def var(v: str) -> str:
    assert v in VARS, f"Unknown variable: {v}"
    return v

def neg(f: str) -> str:
    return f'¬{f}'

def imp(a: str, b: str) -> str:
    return f'({a}→{b})'

# ── Axiom schemas ─────────────────────────────────────────────────────────────

def A1(p: str, q: str) -> str:
    """p → (q → p)"""
    return imp(p, imp(q, p))

def A2(p: str, q: str, r: str) -> str:
    """(p → (q → r)) → ((p → q) → (p → r))"""
    return imp(imp(p, imp(q, r)), imp(imp(p, q), imp(p, r)))

def A3(p: str, q: str) -> str:
    """(¬q → ¬p) → (p → q)"""
    return imp(imp(neg(q), neg(p)), imp(p, q))

# ── Parser ────────────────────────────────────────────────────────────────────

def is_atomic(f: str) -> bool:
    return f in VARS

def is_negation(f: str) -> bool:
    return f.startswith('¬')

def is_implication(f: str) -> bool:
    return f.startswith('(') and f.endswith(')')

def get_neg_body(f: str) -> Optional[str]:
    """¬X → X"""
    if is_negation(f):
        return f[1:]
    return None

def split_implication(f: str) -> Optional[Tuple[str, str]]:
    """
    (A→B) → (A, B)
    Finds the main connective → at depth 0 inside the outer parens.
    """
    if not (f.startswith('(') and f.endswith(')')):
        return None
    inner = f[1:-1]
    depth = 0
    for i, c in enumerate(inner):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '→' and depth == 0:
            return inner[:i], inner[i+1:]
    return None

def formula_depth(f: str) -> int:
    """Max nesting depth of a formula."""
    if is_atomic(f):
        return 0
    if is_negation(f):
        return 1 + formula_depth(get_neg_body(f))
    parts = split_implication(f)
    if parts:
        return 1 + max(formula_depth(parts[0]), formula_depth(parts[1]))
    return 0

def formula_size(f: str) -> int:
    """Number of connectives/atoms."""
    if is_atomic(f):
        return 1
    if is_negation(f):
        return 1 + formula_size(get_neg_body(f))
    parts = split_implication(f)
    if parts:
        return 1 + formula_size(parts[0]) + formula_size(parts[1])
    return 1

def get_vars_in_order(f: str) -> List[str]:
    """Return variables in order of first appearance."""
    seen, result = set(), []
    def walk(f):
        if is_atomic(f):
            if f not in seen:
                seen.add(f); result.append(f)
        elif is_negation(f):
            walk(get_neg_body(f))
        else:
            parts = split_implication(f)
            if parts:
                walk(parts[0]); walk(parts[1])
    walk(f)
    return result

# ── Alpha-equivalence (canonical form) ───────────────────────────────────────

def canonicalize(f: str) -> str:
    """
    Rename variables by order of first appearance:
    first var → p, second → q, third → r.
    This makes (q→q) and (r→r) and (p→p) all equal to canonical "(p→p)".
    """
    orig_vars = get_vars_in_order(f)
    rename = {v: VARS[i] for i, v in enumerate(orig_vars) if i < len(VARS)}

    def subst(f: str) -> str:
        if is_atomic(f):
            return rename.get(f, f)
        if is_negation(f):
            return neg(subst(get_neg_body(f)))
        parts = split_implication(f)
        if parts:
            return imp(subst(parts[0]), subst(parts[1]))
        return f

    return subst(f)

def alpha_eq(f1: str, f2: str) -> bool:
    """Check structural equivalence up to variable renaming."""
    return canonicalize(f1) == canonicalize(f2)

# ── Modus Ponens ──────────────────────────────────────────────────────────────

def find_mp_consequences(proven: Set[str]) -> List[Tuple[str, str, str]]:
    """
    Find all valid MP applications in the proven set.
    Returns list of (antecedent, implication, consequent).
    For each formula (A→B) in proven where A is also in proven.
    """
    results = []
    for f in proven:
        parts = split_implication(f)
        if parts:
            antecedent, consequent = parts
            if antecedent in proven and consequent not in proven:
                results.append((antecedent, f, consequent))
    return results

# ── Checkpoint library ────────────────────────────────────────────────────────
# Defined as canonical forms (variables renamed by first appearance).
# Matching uses alpha_eq.

CHECKPOINT_CANONICAL = {
    'identity':        canonicalize(A1('p', 'p')),            # (p→p) -- but wait this is (p→(p→p))
    # Let's define them properly as the RESULT formula
}

# The actual important theorems (as canonical formula strings):
CHECKPOINTS_DEF: Dict[str, dict] = {
    'T_identity':     {
        'canonical': '(p→p)',
        'name':      'Identity',
        'reward':    8.0,
        'example':   A1('p', 'p'),   # not the proof, just a formula to show
    },
    'T_double_neg':   {
        'canonical': '(¬¬p→p)',
        'name':      'Double Negation Elimination',
        'reward':    12.0,
        'example':   '(¬¬p→p)',
    },
    'T_contrapos':    {
        'canonical': '((p→q)→(¬q→¬p))',
        'name':      'Contrapositive',
        'reward':    10.0,
        'example':   '((p→q)→(¬q→¬p))',
    },
    'T_ex_falso':     {
        'canonical': '(¬p→(p→q))',
        'name':      'Ex Falso Quodlibet',
        'reward':    8.0,
        'example':   '(¬p→(p→q))',
    },
    'T_hyp_syll':     {
        'canonical': '((p→q)→((q→r)→(p→r)))',
        'name':      'Hypothetical Syllogism',
        'reward':    9.0,
        'example':   '((p→q)→((q→r)→(p→r)))',
    },
    'T_peirce':       {
        'canonical': '(((p→q)→p)→p)',
        'name':      "Peirce's Law",
        'reward':    11.0,
        'example':   '(((p→q)→p)→p)',
    },
    'T_weakening':    {
        'canonical': '(p→(q→p))',
        'name':      'Weakening (A1 instance)',
        'reward':    3.0,
        'example':   '(p→(q→p))',
    },
}

def match_checkpoint(f: str) -> Optional[str]:
    """
    If f is alpha-equivalent to a checkpoint formula, return the checkpoint id.
    Otherwise return None.
    """
    c = canonicalize(f)
    for cp_id, cp_def in CHECKPOINTS_DEF.items():
        if c == cp_def['canonical']:
            return cp_id
    return None

def is_interesting(f: str, proven: Set[str]) -> bool:
    """
    Heuristic: is this a formula worth keeping?
    Filters out trivially redundant or too-complex formulas.
    """
    if formula_size(f) > 14:       # too complex for now
        return False
    if formula_depth(f) > 5:
        return False
    if f in proven:
        return False
    return True
