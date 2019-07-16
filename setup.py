# Copyright (C) 2019  UC Berkeley
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import ast
from setuptools import setup
import sys

setup_requires = ['setuptools >= 30.3.0']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires.append('pytest-runner')

# Get docstring and version without importing module
with open('deepCR/__init__.py') as f:
    mod = ast.parse(f.read())
__doc__ = ast.get_docstring(mod)
__version__ = mod.body[-1].value.s

setup(description=__doc__.splitlines()[1],
      long_description=__doc__,
      version=__version__,
      include_package_data=True,
      setup_requires=setup_requires)
