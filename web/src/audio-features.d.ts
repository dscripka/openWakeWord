import type { InferenceSession } from "onnxruntime-web";

export const MEL_BINS: number;
export const EMBED_DIM: number;
export const WINDOW_SIZE: number;
export const STEP_SIZE: number;
export const CHUNK: number;

/** Tensor-ready features: a flat Float32Array plus its dimensions. */
export interface Features {
  data: Float32Array;
  dims: number[];
}

/**
 * Streaming melspectrogram + speech-embedding feature pipeline. Faithful port
 * of `openwakeword.utils.AudioFeatures`. All audio is 16-bit PCM @ 16 kHz.
 */
export class AudioFeatures {
  constructor(
    melspecSession: InferenceSession,
    embeddingSession: InferenceSession,
    opts?: { sampleRate?: number }
  );

  /** Reset all internal streaming buffers. Call `warmup()` afterwards. */
  reset(skipWarmup?: boolean): void;

  /** Seed the feature buffer with embeddings of ~4 s of audio. Await once after construction/reset. */
  warmup(): Promise<void>;

  /**
   * Feed 16-bit PCM @ 16 kHz audio (ideally multiples of 1280 samples).
   * @returns the number of samples processed this call.
   */
  streamingFeatures(x: Int16Array): Promise<number>;

  /** Most recent feature frames as a tensor-ready `{ data, dims }`. */
  getFeatures(nFeatureFrames?: number, startNdx?: number): Features;
}
