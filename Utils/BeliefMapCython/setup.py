try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


from Cython.Build import cythonize
import os


setup(
    ext_modules = cythonize("BeliefMapCython.pyx")
)