# play_app.py
import onnxruntime as ort
import numpy as np
import json

sess = ort.InferenceSession("connect4.onnx")
def predict(board, player):
    current = (board==player).astype(np.float32)
    opponent = (board==(1 if player==2 else 2)).astype(np.float32)
    x = np.stack([current, opponent], axis=0)[None, ...].astype(np.float32)
    out = sess.run(None, {'input': x})[0]
    move = int(np.argmax(out, axis=1)[0])
    print("predicted column", move)
