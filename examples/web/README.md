# Examples

This folder contains examples of using openWakeWord with web applications.

## Websocket Streaming

As openWakeWord does not have a native Javascript port, using it within a web browswer is best accomplished with websocket streaming of the audio data from the browser to a simple Python application. To install the requirements for this example:

```
pip install aiohttp
```

The `streaming_client.html` page shows a simple implementation of audio capture and streamimng from a microphone and streaming in a browser, and the `streaming_server.py` file is the corresponding websocket server that passes the audio into openWakeWord.

To run the example, execute `python streaming_server.py` (add the `--help` argument to see options) and navigate to `localhost:9000` in your browser.