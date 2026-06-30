export interface MicrophoneOptions {
  /**
   * URL of the AudioWorklet module (`mic-worklet.js`). Defaults to the file
   * shipped next to this module. Override it if your bundler does not emit the
   * worklet asset automatically (point it at a copy served same-origin).
   */
  workletUrl?: string;
}

/**
 * Captures microphone audio as 16-bit PCM @ 16 kHz and delivers it in
 * 1280-sample (80 ms) frames via the `onFrame` callback.
 */
export class Microphone {
  constructor(onFrame: (frame: Int16Array) => void, opts?: MicrophoneOptions);

  /** Request mic access and start delivering frames. Requires a secure context. */
  start(): Promise<void>;

  /** Stop capture and release the microphone. */
  stop(): Promise<void>;

  /** Actual sample rate of the capture context (should be 16000), or null if not started. */
  readonly sampleRate: number | null;
}
