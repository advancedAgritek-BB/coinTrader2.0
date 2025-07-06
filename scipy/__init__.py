from types import ModuleType
stats = ModuleType('scipy.stats')
from .stats import pearsonr
stats.pearsonr = pearsonr
__all__ = ['stats']
