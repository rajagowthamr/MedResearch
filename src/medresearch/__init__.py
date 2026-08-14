"""MedResearch: clinical language modelling.

Two independent tracks share this package:

  medresearch.gpt  the from-scratch character-level GPT (model, tokenizer, data)
  medresearch.rf   the Random Forest test-result baseline

Paths for both live in medresearch.config -- import them from there rather than
writing relative filenames, so scripts work from any working directory.
"""

__version__ = "0.5.0"
