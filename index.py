import os
import wave
import torch
import contextlib
from flask import Flask, request

# create the Flask app
app = Flask(__name__)

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)


@app.route('/')
def query_example():
    # if key doesn't exist, returns None
    language = request.args.get('text')
    fileid = request.args.get('id')
    example_text = format(language)
    sample_rate = 48000
    speaker = 'aidar'

    audio = model.apply_tts(text=example_text,
                            speaker=speaker,
                            sample_rate=sample_rate)

    def write_wave(path, audio, sample_rate):
        """Writes a .wav file.
        Takes path, PCM audio data, and sample rate.
        """
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    templ = 'static/wav/' + fileid + '.mp3'
    write_wave(path=format(templ),
               audio=(audio * 32767).numpy().astype('int16'),
               sample_rate=sample_rate)

    print(model.speakers)

    return '''
    <audio controls>
      <source
        src="{}"
        type="audio/mpeg"
      />'''.format(os.path.join(app.root_path, 'static/wav/', format(templ)))


@app.route('/form-example')
def form_example():
    return 'Form Data Example'


@app.route('/json-example')
def json_example():
    return 'JSON Object Example'


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=2000)
