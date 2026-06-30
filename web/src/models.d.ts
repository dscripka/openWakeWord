/** Feature models required by every wake word model. */
export const FEATURE_MODELS: { melspectrogram: string; embedding: string };

/** Voice-activity-detection models (not yet used by the browser port). */
export const VAD_MODELS: { silero_vad: string };

/** Pre-trained wake word models, keyed by name -> ONNX filename. */
export const PRETRAINED_MODELS: Record<string, string>;

/** Integer-class -> label mappings for multi-class models (e.g. `timer`). */
export const MODEL_CLASS_MAPPINGS: Record<string, Record<number, string>>;
