// Helper to capture microphone audio as 16-bit PCM @ 16 kHz frames and feed
// them to a callback. Uses an AudioWorklet (see mic-worklet.js).

export class Microphone {
  /**
   * @param {(frame: Int16Array) => void} onFrame called with 1280-sample frames
   * @param {{workletUrl?: string}} [opts]
   */
  constructor(onFrame, opts = {}) {
    this.onFrame = onFrame;
    this.workletUrl =
      opts.workletUrl ?? new URL("./mic-worklet.js", import.meta.url).href;
    this.context = null;
    this.stream = null;
    this.node = null;
    this.source = null;
  }

  async start() {
    if (this.context) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Prefer a 16 kHz context (the worklet also resamples as a fallback in
    // case the browser ignores or rejects this hint).
    const AudioCtx = globalThis.AudioContext || globalThis.webkitAudioContext;
    try {
      this.context = new AudioCtx({ sampleRate: 16000 });
    } catch {
      this.context = new AudioCtx();
    }
    if (this.context.state === "suspended") await this.context.resume();
    await this.context.audioWorklet.addModule(this.workletUrl);

    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, "pcm-worklet");
    this.node.port.onmessage = (e) => this.onFrame(e.data);

    this.source.connect(this.node);
    // Keep the worklet pulling audio without routing mic to the speakers.
    const sink = this.context.createGain();
    sink.gain.value = 0;
    this.node.connect(sink);
    sink.connect(this.context.destination);
    this._sink = sink;
  }

  /** Actual sample rate of the capture context (should be 16000). */
  get sampleRate() {
    return this.context ? this.context.sampleRate : null;
  }

  async stop() {
    if (this.node) this.node.port.onmessage = null;
    try {
      this.source?.disconnect();
      this.node?.disconnect();
      this._sink?.disconnect();
    } catch {}
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.context) await this.context.close();
    this.context = null;
    this.stream = null;
    this.node = null;
    this.source = null;
  }
}
