#!/usr/bin/python
# coding=utf-8
from distutils.core import setup

setup(name='PyLit',
      # version='0.7.9',
      description='Literate programming with reStructuredText',
      long_description="""

PyLit (Python Literate) provides a plain but efficient tool for `literate
programming`_: a bidirectional text/code converter. 

   The idea is that you do not document programs (after the fact), but
   write documents that *contain* the programs.
   
   -- John Max Skaller in a `Charming Python interview`_

Features
--------

* `Dual Source`_
* Simplicity
* Markup with reStructuredText_
* Python Doctest Support

.. _Charming Python interview:
    http://www.ibm.com/developerworks/library/l-pyth7.html
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _dual source: http://pylit.berlios.de/features.html#dual-source
.. _literate programming: http://pylit.berlios.de/literate-programming.html
      """,
      author='Guenter Milde',
      author_email='milde@users.sf.net',
      url='http://pylit.berlios.de/',
      download_url='http://pylit.berlios.de/download/',
      classifiers=[
                   'Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Natural Language :: English',
                   'Natural Language :: German',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.4',
                   'Topic :: Software Development :: Documentation',
                   'Topic :: Software Development :: User Interfaces',
                   'Topic :: Text Processing :: Markup'
                  ],
      provides=['pylit'],
      scripts=['rstdocs/download/pylit'],
      package_dir = {'': 'src'},
      py_modules = ['pylit']
     )
