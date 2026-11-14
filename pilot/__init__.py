# -*- coding: utf-8 -*-
"""AlgoPilot:自驱动搜广推算法迭代智能组织。

引擎(pilot.*)域无关;算法域(domains/*)通过 domain.json 契约接入。
"""
__version__ = "0.1.0"

from .domain import DomainSpec  # noqa: F401
from .memory import Ledger      # noqa: F401
