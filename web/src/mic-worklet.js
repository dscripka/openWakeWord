// AudioWorklet processor that converts the microphone stream into 16-bit PCM
// frames of 1280 samples (80 ms @ 16 kHz) and posts them to the main thread.
//
// It resamples from the AudioContext's native rate (the global `sampleRate`
// inside the worklet scope) down/up to 16 kHz using linear interpolation, so
// it works even when the browser ignores the requested 16 kHz context rate.

const TARGET_RATE = 16000;
const FRAME = 1280;

class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ratio = sampleRate / TARGET_RATE; // input samples per output sample
    this._buf = new Int16Array(FRAME);
    this._n = 0;
    this._tail = new Float32Array(0); // leftover input samples between blocks
    this._frac = 0; // fractional read position within the current data buffer
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    // Prepend any leftover samples needed for cross-block interpolation.
    let data = channel;
    if (this._tail.length) {
      data = new Float32Array(this._tail.length + channel.length);
      data.set(this._tail, 0);
      data.set(channel, this._tail.length);
    }

    const ratio = this._ratio;
    let t = this._frac;
    while (Math.floor(t) + 1 < data.length) {
      const i = Math.floor(t);
      const frac = t - i;
      const s = data[i] + (data[i + 1] - data[i]) * frac; // linear interp
      let v = Math.floor(32767 * s);
      if (v > 32767) v = 32767;
      else if (v < -32768) v = -32768;
      this._buf[this._n++] = v;
      if (this._n === FRAME) {
        this.port.postMessage(this._buf.slice());
        this._n = 0;
      }
      t += ratio;
    }

    const keepFrom = Math.floor(t);
    this._tail = data.slice(keepFrom);
    this._frac = t - keepFrom;
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
