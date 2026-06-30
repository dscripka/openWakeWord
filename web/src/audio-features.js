import * as ort from "onnxruntime-web";

// Faithful JavaScript port of `openwakeword.utils.AudioFeatures`.
//
// Implements the streaming melspectrogram + Google speech_embedding feature
// pipeline. All audio is 16-bit PCM @ 16 kHz, exactly like the Python library.
//
//   raw int16 audio  ->  melspectrogram model  ->  (frames x 32) mel features
//                    ->  embedding model        ->  (frames x 96) embeddings
//
// The embeddings are what the wake word models consume.

const MEL_BINS = 32;
const EMBED_DIM = 96;
const WINDOW_SIZE = 76; // mel frames per embedding window
const STEP_SIZE = 8; // mel frames produced per 1280-sample (80 ms) chunk
const CHUNK = 1280; // samples per processing step (80 ms @ 16 kHz)

export class AudioFeatures {
  /**
   * @param {ort.InferenceSession} melspecSession
   * @param {ort.InferenceSession} embeddingSession
   */
  constructor(melspecSession, embeddingSession, { sampleRate = 16000 } = {}) {
    this.melspecSession = melspecSession;
    this.embeddingSession = embeddingSession;
    this.sampleRate = sampleRate;

    this.melspecInputName = melspecSession.inputNames[0]; // "input"
    this.embeddingInputName = embeddingSession.inputNames[0]; // "input_1"

    this.rawDataMaxLen = sampleRate * 10;
    this.melspectrogramMaxLen = 10 * 97; // ~10 s of mel frames
    this.featureBufferMaxLen = 120; // ~10 s of embedding history

    this.reset(/* skipWarmup */ true);
  }

  /** Reset all internal streaming buffers. */
  reset(skipWarmup = false) {
    this.rawDataBuffer = []; // numbers (int16 samples)
    this.rawDataRemainder = new Int16Array(0);
    this.accumulatedSamples = 0;
    // Mel buffer is seeded with ones, matching np.ones((76, 32)).
    this.melBuffer = [];
    for (let i = 0; i < WINDOW_SIZE; i++) {
      this.melBuffer.push(new Float32Array(MEL_BINS).fill(1));
    }
    // Feature buffer is seeded later by warmup() (needs async model calls).
    this.featureBuffer = [];
    if (!skipWarmup) {
      // synchronous reset cannot run the models; callers should await warmup()
    }
  }

  /**
   * Seed the feature buffer with the embeddings of 4 s of (random) audio, as
   * the Python implementation does on init/reset. Must be awaited once after
   * construction (and after reset()).
   */
  async warmup() {
    const audio = new Int16Array(this.sampleRate * 4);
    for (let i = 0; i < audio.length; i++) {
      audio[i] = Math.floor(Math.random() * 2000 - 1000);
    }
    this.featureBuffer = await this._getEmbeddings(audio);
  }

  // --- melspectrogram -------------------------------------------------------

  /**
   * Compute the melspectrogram of int16 audio.
   * Returns an array of Float32Array rows, shape (frames, 32), with the
   * `x / 10 + 2` transform applied (matches the Python default).
   * @param {Int16Array} int16
   */
  async _getMelspectrogram(int16) {
    const x = Float32Array.from(int16); // int16 magnitudes as float (NOT normalized)
    const tensor = new ort.Tensor("float32", x, [1, x.length]);
    const out = await this.melspecSession.run({ [this.melspecInputName]: tensor });
    const o = out[this.melspecSession.outputNames[0]];
    const dims = o.dims;
    const bins = dims[dims.length - 1];
    const frames = dims[dims.length - 2];
    const data = o.data;
    const rows = [];
    for (let f = 0; f < frames; f++) {
      const row = new Float32Array(bins);
      const base = f * bins;
      for (let b = 0; b < bins; b++) {
        row[b] = data[base + b] / 10 + 2;
      }
      rows.push(row);
    }
    return rows;
  }

  // --- embeddings -----------------------------------------------------------

  /**
   * Run the embedding model over a batch of 76x32 mel windows.
   * @param {Float32Array[][]} windows array of windows, each window is 76 rows of 32
   * @returns {Promise<Float32Array[]>} array of 96-dim embedding rows
   */
  async _embedWindows(windows) {
    const n = windows.length;
    if (n === 0) return [];
    const data = new Float32Array(n * WINDOW_SIZE * MEL_BINS);
    let p = 0;
    for (const win of windows) {
      for (let r = 0; r < WINDOW_SIZE; r++) {
        data.set(win[r], p);
        p += MEL_BINS;
      }
    }
    const tensor = new ort.Tensor("float32", data, [n, WINDOW_SIZE, MEL_BINS, 1]);
    const out = await this.embeddingSession.run({
      [this.embeddingInputName]: tensor,
    });
    const o = out[this.embeddingSession.outputNames[0]];
    const flat = o.data; // length n * 96
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push(flat.slice(i * EMBED_DIM, i * EMBED_DIM + EMBED_DIM));
    }
    return rows;
  }

  /**
   * Compute embeddings for a whole audio clip (used for warmup / batch use).
   * @param {Int16Array} int16
   */
  async _getEmbeddings(int16) {
    const spec = await this._getMelspectrogram(int16);
    const windows = [];
    for (let i = 0; i < spec.length; i += STEP_SIZE) {
      const window = spec.slice(i, i + WINDOW_SIZE);
      if (window.length === WINDOW_SIZE) windows.push(window);
    }
    return this._embedWindows(windows);
  }

  // --- streaming ------------------------------------------------------------

  _bufferRawData(int16) {
    for (let i = 0; i < int16.length; i++) this.rawDataBuffer.push(int16[i]);
    if (this.rawDataBuffer.length > this.rawDataMaxLen) {
      this.rawDataBuffer = this.rawDataBuffer.slice(-this.rawDataMaxLen);
    }
  }

  async _streamingMelspectrogram(nSamples) {
    if (this.rawDataBuffer.length < 400) {
      throw new Error(
        "The number of input frames must be at least 400 samples @ 16khz (25 ms)!"
      );
    }
    const start = Math.max(0, this.rawDataBuffer.length - (nSamples + 160 * 3));
    const tail = Int16Array.from(this.rawDataBuffer.slice(start));
    const rows = await this._getMelspectrogram(tail);
    for (const row of rows) this.melBuffer.push(row);
    if (this.melBuffer.length > this.melspectrogramMaxLen) {
      this.melBuffer = this.melBuffer.slice(-this.melspectrogramMaxLen);
    }
  }

  /**
   * Streaming feature extraction. Feed it 16-bit PCM @ 16 kHz audio frames
   * (ideally multiples of 1280 samples / 80 ms).
   * @param {Int16Array} x
   * @returns {Promise<number>} number of samples that were processed this call
   */
  async streamingFeatures(x) {
    let processedSamples = 0;

    if (this.rawDataRemainder.length !== 0) {
      const merged = new Int16Array(this.rawDataRemainder.length + x.length);
      merged.set(this.rawDataRemainder, 0);
      merged.set(x, this.rawDataRemainder.length);
      x = merged;
      this.rawDataRemainder = new Int16Array(0);
    }

    if (this.accumulatedSamples + x.length >= CHUNK) {
      const remainder = (this.accumulatedSamples + x.length) % CHUNK;
      if (remainder !== 0) {
        const xEven = x.subarray(0, x.length - remainder);
        this._bufferRawData(xEven);
        this.accumulatedSamples += xEven.length;
        this.rawDataRemainder = x.slice(x.length - remainder);
      } else {
        this._bufferRawData(x);
        this.accumulatedSamples += x.length;
        this.rawDataRemainder = new Int16Array(0);
      }
    } else {
      this.accumulatedSamples += x.length;
      this._bufferRawData(x);
    }

    if (this.accumulatedSamples >= CHUNK && this.accumulatedSamples % CHUNK === 0) {
      await this._streamingMelspectrogram(this.accumulatedSamples);

      // Compute new embeddings for each newly-available 80 ms chunk.
      for (let i = this.accumulatedSamples / CHUNK - 1; i >= 0; i--) {
        const ndx = -STEP_SIZE * i === 0 ? this.melBuffer.length : -STEP_SIZE * i;
        const endAbs = ndx < 0 ? this.melBuffer.length + ndx : ndx;
        const startAbs = endAbs - WINDOW_SIZE;
        if (startAbs >= 0) {
          const window = this.melBuffer.slice(startAbs, endAbs);
          const [embedding] = await this._embedWindows([window]);
          this.featureBuffer.push(embedding);
        }
      }

      processedSamples = this.accumulatedSamples;
      this.accumulatedSamples = 0;
    }

    if (this.featureBuffer.length > this.featureBufferMaxLen) {
      this.featureBuffer = this.featureBuffer.slice(-this.featureBufferMaxLen);
    }

    return processedSamples !== 0 ? processedSamples : this.accumulatedSamples;
  }

  /**
   * Return the most recent feature frames as a tensor-ready {data, dims}.
   * Mirrors `AudioFeatures.get_features`.
   * @param {number} nFeatureFrames
   * @param {number} startNdx negative index into the feature buffer, or -1
   */
  getFeatures(nFeatureFrames = 16, startNdx = -1) {
    let frames;
    if (startNdx !== -1) {
      const end =
        startNdx + nFeatureFrames === 0 ? undefined : startNdx + nFeatureFrames;
      frames = this.featureBuffer.slice(startNdx, end);
    } else {
      frames = this.featureBuffer.slice(-nFeatureFrames);
    }
    const n = frames.length;
    const data = new Float32Array(n * EMBED_DIM);
    for (let i = 0; i < n; i++) data.set(frames[i], i * EMBED_DIM);
    return { data, dims: [1, n, EMBED_DIM] };
  }
}

export { MEL_BINS, EMBED_DIM, WINDOW_SIZE, STEP_SIZE, CHUNK };
