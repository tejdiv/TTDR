# JAX 0.4.20 compat: jax.tree (with short names like .map, .leaves)
# was added in JAX 0.4.25. Shim it for older versions.
import jax
if not hasattr(jax, "tree"):
    class _TreeShim:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        flatten = staticmethod(jax.tree_util.tree_flatten)
        unflatten = staticmethod(jax.tree_util.tree_unflatten)
        structure = staticmethod(jax.tree_util.tree_structure)
    jax.tree = _TreeShim
