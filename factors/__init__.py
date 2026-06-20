"""Factor library: vet a factor's predictive power (IC, quantiles) before building.

Importing this package registers the built-in factors so the IC harness can score
them automatically.
"""

from factors import fundamentals, library  # noqa: F401  (registers factors on import)
