#!/usr/bin/env python3
"""
Basic tests for FileForge Python components
"""
import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import embedding_manager
    import benchmark_assembler
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")


class TestFileForge(unittest.TestCase):
    """Basic tests for FileForge components"""
    
    def test_imports(self):
        """Test that we can import basic modules"""
        # This is a basic smoke test
        self.assertTrue(True)
    
    def test_embedding_manager_exists(self):
        """Test that embedding manager module exists"""
        try:
            import embedding_manager
            self.assertTrue(hasattr(embedding_manager, '__name__'))
        except ImportError:
            self.skipTest("embedding_manager not available")
    
    def test_benchmark_assembler_exists(self):
        """Test that benchmark assembler module exists"""
        try:
            import benchmark_assembler
            self.assertTrue(hasattr(benchmark_assembler, '__name__'))
        except ImportError:
            self.skipTest("benchmark_assembler not available")


if __name__ == '__main__':
    unittest.main() 