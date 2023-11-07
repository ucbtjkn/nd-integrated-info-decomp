"""Sphinx configuration file for phyid documentation."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'phyid'
copyright = '2023, phyid developers'
author = 'Eric Ceballos, Zhen-Qi Liu, Andrea Luppi, Pedro Mediano, Fernando Rosas'

# Import project to get version info
sys.path.insert(0, os.path.abspath(os.path.pardir))
import phyid  # noqa
# The short X.Y version
version = phyid.__version__
# The full version, including alpha/beta/rc tags
release = phyid.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_default_options = {'members': True, 'inherited-members': True}
numpydoc_show_class_members = False
autoclass_content = "class"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_rtd_theme  # noqa
html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False
# html_logo = '_static/logo.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {'logo_only': True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# CSS files to include
html_css_files = ['theme_overrides.css']

# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org', None),
    'matplotlib': ('https://matplotlib.org', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
}