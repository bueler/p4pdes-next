import re

literal = """\
::
  ::
t ::
text::
 indented::
 indented ::
more text :: 
 indented text :: 
. no-directive::
a .. directive:: somewhere::
"""

directive = """\
.. code-block:: python
  .. code-block:: python
.. code-block:: python listings
  .. code-block:: python listings
"""

misses = """\
.. comment string ::
.. ::
text:
"""

def get_regexp(marker):
    class self: pass   # dummy to make the definitions compatible to pylit
    if marker == '::':
        self.marker_regexp = re.compile('^( *(?!\.\.).*)(%s)([ \n]*)$' 
                                        % marker)
    else:
        # assume code_block_marker is a directive like '.. code-block::'
        self.marker_regexp = re.compile('^( *)(%s)(.*\n?)$' % marker)
    return self.marker_regexp

for marker in ('::', '.. code-block::'):
    print 'regexp test for %r' % marker
    regexp = get_regexp(marker)
    for sample in (literal + directive + misses).splitlines(True):
        match = regexp.search(sample)
        print '%-40r'%(sample), 
        if match:
            print '-> ',  match.groups()
            # print '-> ',  repr(match.group())
        else:
            print '-> ', match

options = """\
  :lineno:
  :lineno: 2
  :line-no:
  :line+no:
  :lineno+:
  :x:x:
"""

no_options = ['  :lineno:2', # no space before option arg
              ':lineno:',    # no leading whitespace
              '  ::',        # empty option
              '  :lin$no:',  # invalid character 
             ]
option_regexp = re.compile(r' +:(\w|[-._+:])+:( |$)')

print 'regexp test for option_regexp'
for sample in (options).splitlines(True) + no_options:
        match = option_regexp.search(sample)
        print '%-40r'%(sample), 
        if match:
            print '-> ',  match.groups()
            # print '-> ',  repr(match.group())
        else:
            print '-> ', match




           
