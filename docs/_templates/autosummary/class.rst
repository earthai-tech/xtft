{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __call__
   :exclude-members: forward, extra_repr

.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Examples using ``{{ objname }}``
