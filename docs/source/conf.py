 #############################################################################
 #                            _     _     _     _                            #
 #                           / \   / \   / \   / \                           #
 #                          ( M ) ( O ) ( M ) ( A )                          #
 #                           \_/   \_/   \_/   \_/                           #
 #                                                                           #
 #               MOMA: Ground vehicles mobility mapping toolkit              #
 #############################################################################

import subprocess


def get_git_revision_short_hash() -> str:
    """Get the short hash of the Git repository in the parent folder."""
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                   cwd="../").decode('ascii').strip()


# Sphinx configuration
project = "MOMA"
copyright = "2024, Aarhus Robotics"
author = "Dario Sirangelo"
release = "1.0.0"
extensions = [
    'myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.graphviz',
    'sphinx.ext.todo'
]
templates_path = ["_templates"]

# Language configuration
primary_domain = 'cpp'
highlight_language = 'cpp'

# HTML build configuration
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_title = "MOMA"
html_logo = "logo.png"
html_context = {"default_mode": "dark"}
html_theme_options = {
    "show_nav_level": 2,
    "show_prev_next": False,
    "collapse_navigation": False
}
html_show_sourcelink = False

# LaTeX build configuration
latex_elements = {
    'extraclassoptions': 'openany,oneside',
    "releasename": "Commit hash",
}
latex_authors = r"Dario Sirangelo"
latex_documents = [("index", "moma.tex", project, latex_authors, "manual")]
pdf_documents = [("index", "moma", "title", latex_authors)]

# Autodoc extension configuration
autodoc_default_options = {
    "special-members": "__init__",
    'members': True,
    'private-members': True,
    'exclude-members': '__weakref__'
}

# Graphviz extension configuration
graphviz_output_format = 'svg'

# Todo extension configuration
todo_include_todos = True

# MyST extension configuration
myst_enable_extensions = [
    "amsmath", "attrs_inline", "colon_fence", "deflist", "dollarmath",
    "fieldlist", "html_admonition", "html_image", "replacements", "smartquotes",
    "strikethrough", "substitution", "tasklist"
]
