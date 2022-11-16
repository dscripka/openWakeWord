# Synthetic Data Generation

The use of synthetic data for training STT or wakeword/phrase detection models is not a new concept, and in particular the inspiration for this library was motivated by several specific papers:

1) Paper 1 (end-to-end SLU)
2) Paper 2 (end-to-end SLU with synthetic)

# Choosing TTS Models

- Focus on variability in the generation (so sampling models)
- Focus on multi-speaker TTS based on speaker embeddings

# Increasing Diversity in Generated Speech

- Use relatively high values for sampling parameters (even if this causes low quality generations some small percentage of the time)
- Use spherical interpolation of embeddings to generate new speakers