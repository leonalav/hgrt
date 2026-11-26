"""Static analysis of the notebook: every name must be defined before use.

Colab executes cells top to bottom in one namespace, so a name that is read at
module level before any cell defines it is a NameError waiting to happen.

This module walks the notebook in execution order and reports:
  * names read at module level that no earlier cell defines
  * names read at module level that NO cell defines
  * function bodies that read a name which is neither a local, a parameter,
    a builtin, nor defined anywhere in the notebook

The check is deliberately conservative: it ignores names bound by `global`,
`nonlocal`, comprehensions, and lambdas, which have their own scopes.
"""

import ast
import builtins

BUILTINS = set(dir(builtins)) | {
    '__name__', '__doc__', '__file__', '__builtins__', '__annotations__',
}


def _bound_names(target):
    """Every name bound by an assignment target, including tuple unpacking."""
    out = set()
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            out |= _bound_names(element)
    elif isinstance(target, ast.Starred):
        out |= _bound_names(target.value)
    return out


def _cell_bound_names(tree):
    """Names a cell binds at module level (visible to later cells)."""
    bound = set()

    def visit_body(body):
        for node in body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    bound.update(_bound_names(target))
            elif isinstance(node, ast.AnnAssign):
                bound.update(_bound_names(node.target))
            elif isinstance(node, ast.AugAssign):
                bound.update(_bound_names(node.target))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, ast.Global):
                bound.update(node.names)
            elif isinstance(node, ast.Nonlocal):
                bound.update(node.names)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                bound.update(_bound_names(node.target))
                visit_body(node.body)
                visit_body(node.orelse)
            elif isinstance(node, (ast.While, ast.If)):
                visit_body(node.body)
                visit_body(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        bound.update(_bound_names(item.optional_vars))
                visit_body(node.body)
            elif isinstance(node, ast.Try):
                visit_body(node.body)
                for handler in node.handlers:
                    if handler.name:
                        bound.add(handler.name)
                    visit_body(handler.body)
                visit_body(node.orelse)
                visit_body(node.finalbody)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    bound.add(node.name)
                visit_body(node.body)

    visit_body(tree.body)
    return bound


def _cell_read_names(tree):
    """Names read at module level (not inside a nested function scope).

    Comprehensions and lambdas have their own scope in Python 3, so their
    target variables are not module-level reads.
    """
    reads = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # A class body reads names, but its methods have their own scope.
                if isinstance(child, ast.ClassDef):
                    visit(child)
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                # Only the outermost iterable is evaluated in the enclosing
                # scope; the rest lives in the comprehension's own scope.
                for gen in child.generators[1:]:
                    visit(gen.iter)
                if child.generators:
                    visit(child.generators[0].iter)
                continue
            if isinstance(child, ast.Lambda):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                reads.append((child.id, child.lineno))
            else:
                visit(child)

    visit(tree)
    return reads


def _function_unresolved(model, tree):
    """Names read inside a function that resolve to nothing known.

    A function body may legitimately read a name defined by a later cell, as
    long as the function is not called before that cell runs. So this returns
    candidates, and the caller decides.

    Enclosing function scopes are honoured: a nested function may capture a
    name from the function that contains it.
    """
    problems = []

    def local_names(func):
        """Names bound inside the function body."""
        assigned = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assigned.update(_bound_names(target))
            elif isinstance(node, ast.AnnAssign):
                assigned.update(_bound_names(node.target))
            elif isinstance(node, ast.AugAssign):
                assigned.update(_bound_names(node.target))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                assigned.update(_bound_names(node.target))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        assigned.update(_bound_names(item.optional_vars))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assigned.add((alias.asname or alias.name).split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assigned.add(alias.asname or alias.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                assigned.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assigned.add(node.name)
        return assigned

    def params_of(func):
        names = {a.arg for a in func.args.args}
        names |= {a.arg for a in func.args.kwonlyargs}
        names |= {a.arg for a in func.args.posonlyargs}
        if func.args.vararg:
            names.add(func.args.vararg.arg)
        if func.args.kwarg:
            names.add(func.args.kwarg.arg)
        return names

    def walk_func(func, enclosing):
        """Check one function, then recurse into the functions it defines."""
        known = params_of(func) | local_names(func) | enclosing | model.known | BUILTINS

        # Only check names in THIS function's own body, excluding the bodies
        # of functions it defines (those are checked separately with this
        # function's scope as their enclosing scope).
        for node in ast.walk(func):
            if node is func:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in known:
                    problems.append((func.name, node.id, node.lineno))

        for node in ast.walk(func):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func:
                walk_func(node, known)

    for func in [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        walk_func(func, set())

    return problems


class NotebookScope:
    """Tracks what is defined as the notebook executes cell by cell."""

    def __init__(self):
        self.known = set()
        self.order = []

    def enter(self, cell_id, tree):
        """Fold one cell into the scope; return reads that were undefined.

        A cell's own definitions are visible to the rest of that cell, so the
        cell's bound names are merged in BEFORE its reads are checked.
        """
        reads = _cell_read_names(tree)
        self.known |= _cell_bound_names(tree)
        undefined = []
        for name, lineno in reads:
            if name in self.known or name in BUILTINS:
                continue
            undefined.append((name, lineno))

        self.order.append((cell_id, tree))
        return undefined


def analyse(nb):
    """Analyse a parsed notebook.

    Returns a dict with:
      undefined_reads  - [(cell_id, name, lineno)] read at module level with
                          nothing defined before it
      unknown_globals  - [(cell_id, name, lineno)] read at module level and
                          defined nowhere in the notebook
      unresolved_funcs - [(cell_id, func, name, lineno)] names read inside a
                          function that resolve to nothing known
    """
    scope = NotebookScope()
    undefined_reads = []
    trees = []
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        cid = cell.get('id', '?')
        trees.append((cid, tree))
        for name, lineno in scope.enter(cid, tree):
            undefined_reads.append((cid, name, lineno))

    # Anything still unknown after the whole notebook ran is never defined.
    unknown_globals = []
    for cid, name, lineno in undefined_reads:
        if name not in scope.known:
            unknown_globals.append((cid, name, lineno))

    unresolved_funcs = []
    for cid, tree in trees:
        for func, name, lineno in _function_unresolved(scope, tree):
            unresolved_funcs.append((cid, func, name, lineno))

    return dict(undefined_reads=undefined_reads,
                unknown_globals=unknown_globals,
                unresolved_funcs=unresolved_funcs,
                defined=sorted(scope.known))
