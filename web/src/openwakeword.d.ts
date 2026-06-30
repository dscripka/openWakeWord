import { AudioFeatures } from "./audio-features.js";

export * from "./models.js";
export { AudioFeatures } from "./audio-features.js";

/** Options for {@link configureOrt}. */
export interface ConfigureOrtOptions {
  /** Base URL/path for the ONNX Runtime Web wasm binaries. */
  wasmPaths?: string;
  /** Number of wasm threads. Use 1 to avoid requiring COOP/COEP headers. */
  numThreads?: number;
  /** Enable SIMD wasm. */
  simd?: boolean;
}

/**
 * Configure the ONNX Runtime Web environment. Call once before
 * {@link OpenWakeWord.create} to point at self-hosted wasm or tweak threading.
 */
export function configureOrt(opts?: ConfigureOrtOptions): void;

/** A custom wake word model supplied by URL. */
export interface CustomWakewordModel {
  name: string;
  url: string;
  inputFrames?: number;
  classMapping?: Record<number, string>;
}

/** Wake word model reference: a pre-trained name, or a custom model by URL. */
export type WakewordModelRef = string | CustomWakewordModel;

/** Payload passed to {@link OpenWakeWordOptions.onDetection}. */
export interface DetectionEvent {
  /** The wake word label that was detected. */
  label: string;
  /** Detection score in 0..1. */
  score: number;
}

/** Options for {@link OpenWakeWord.create}. */
export interface OpenWakeWordOptions {
  /** Base URL/path for model files. Default `"./models/"`. */
  baseUrl?: string;
  /**
   * Wake word models to load. Strings are looked up in the pre-trained registry
   * (e.g. `"hey_jarvis"`); objects load a custom model by URL. Defaults to all
   * pre-trained models.
   */
  wakewordModels?: WakewordModelRef[];
  /** Override the melspectrogram model URL. */
  melspectrogramUrl?: string;
  /** Override the embedding model URL. */
  embeddingUrl?: string;
  /** ONNX Runtime execution providers. Default `["wasm"]`. */
  executionProviders?: string[];
  /** Options forwarded to {@link configureOrt}. */
  ort?: ConfigureOrtOptions;
  /**
   * Score threshold for triggering {@link onDetection}. Default `0.5`.
   * Also stored as `oww.threshold` so it can be changed at runtime.
   */
  threshold?: number;
  /**
   * Called from within {@link OpenWakeWord.predict} whenever a label's score
   * meets or exceeds {@link threshold}. May be called multiple times per
   * `predict()` invocation if several labels fire simultaneously.
   */
  onDetection?: (event: DetectionEvent) => void;
}

/**
 * Native browser port of `openwakeword.Model`. Runs the full melspectrogram ->
 * embedding -> wake word pipeline client-side using ONNX Runtime Web.
 */
export class OpenWakeWord {
  features: AudioFeatures;
  /** Detection score threshold. Can be updated at runtime. Default `0.5`. */
  threshold: number;
  /** Callback fired on detection. Can be replaced at runtime. */
  onDetection: ((event: DetectionEvent) => void) | null;

  /** Create and initialise a model. */
  static create(opts?: OpenWakeWordOptions): Promise<OpenWakeWord>;

  /** Names of the loaded wake word models. */
  readonly modelNames: string[];

  /** Reset all streaming/prediction state. */
  reset(): Promise<void>;

  /**
   * Predict wake word scores for a frame of 16-bit PCM @ 16 kHz audio (ideally
   * multiples of 1280 samples / 80 ms).
   * @returns a `{ label: score }` map, score in 0..1.
   */
  predict(x: Int16Array): Promise<Record<string, number>>;
}

export default OpenWakeWord;
