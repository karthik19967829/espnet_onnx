from espnet_onnx.export import ASRModelExport

tag_name = 'espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp'

m = ASRModelExport()
transducer_config = {
	'search_type': 'default',
    'score_norm': True
}

# You can export with export_from_pretrained method
m.export_from_pretrained(
    tag_name,
    pretrained_config={
        'beam_size': 20,
        'transducer_conf': transducer_config
    }
)

# or you can export from loaded model
from espnet2.bin.asr_inference import Speech2Text
st = Speech2Text.from_pretrained(
    tag_name,
    beam_size=20,
    transducer_conf=transducer_config,
    device='cpu' # you need to specify device=cpu for exportation.
)
m.export(st, 'transducer_model', quantize=False, optimize=False)