import os
from unittest import TestCase

from mango.tests.const import normalize_path

import mango


class InitializeTest(TestCase):
    def test_version_matches(self):
        """
        Test that the version is properly set (dynamic version from setuptools_scm)
        """
        # Verificar que la versión existe y no es None
        self.assertIsNotNone(mango.__version__)
        self.assertIsInstance(mango.__version__, str)
        self.assertGreater(len(mango.__version__), 0)

        # Verificar que la versión tiene formato válido (X.Y.Z)
        version_parts = mango.__version__.split(".")
        self.assertGreaterEqual(len(version_parts), 2)

        # Verificar que las partes son números o versiones pre-release válidas
        for part in version_parts[:2]:  # Solo las primeras dos partes
            self.assertTrue(
                part.isdigit()
                or part.endswith("a")
                or part.endswith("b")
                or part.endswith("rc")
            )
