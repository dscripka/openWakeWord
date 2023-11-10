# Examples

This folder contains examples of using openWakeWord with web applications.

## Websocket Streaming

As openWakeWord does not have a native Javascript port, using it within a web browswer is best accomplished with websocket streaming of the audio data from the browser to a simple Python application. To install the requirements for this example:

```
pip install aiohttp
pip install resampy
```

The `streaming_client.html` page shows a simple implementation of audio capture and streamimng from a microphone and streaming in a browser, and the `streaming_server.py` file is the corresponding websocket server that passes the audio into openWakeWord.

To run the example, execute `python streaming_server.py` (add the `--help` argument to see options) and navigate to `localhost:9000` in your browser.

Note that this example is illustrative only, and integration of this approach with other web applications may have different requirements. In particular, some key considerations:

- This example captures PCM audio from the web browser and streams full 16-bit integer representations of ~250 ms audio chunks over the websocket connection. In practice, bandwidth efficient streams of compressed audio may be more suitable for some applications.
- The browser captures audio at the native sampling rate of the capture device, which can require re-sampling prior to passing the audio data to openWakeWord. This example uses the `resampy` library which has a good balance between performance and quality, but other resampling approaches that optimize different aspects may be more suitable for some applications.