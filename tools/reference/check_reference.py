"""Check the reference against the upstream encodec and vocos packages.

1. Streaming encoder against upstream with zero padding (the streaming start
   state): latents must match to float tolerance, indices near exactly.
2. Streaming encoder against upstream as released (reflect padding at the
   start of the file): where the indices differ.
3. Whole-signal reference decoder against the vocos package.
4. Streaming reference decoder (with flush) against its whole-signal
   definition, and full lookahead against Vocos.
"""
import numpy as np
import torch
from encodec import EncodecModel
from encodec.modules.conv import SConv1d
from vocos import Vocos

import common
import vocoscodec_ref as ref

torch.set_num_threads(8)
t = ref.load(common.MODEL_DIR, half_weights=False)  # upstream is float32
clip = common.corpus("dev")[0]
x16, _ = common.read_wav(clip)
x = common.up_16k_to_24k(x16)
x = x[: len(x) // 320 * 320]
frames = len(x) // 320

upstream = EncodecModel.encodec_model_24khz().eval()
upstream.set_target_bandwidth(12.0)
with torch.no_grad():
    reflect_latent = upstream.encoder(torch.from_numpy(x)[None, None])[0]
    reflect_codes = upstream.quantizer.encode(reflect_latent[None], 75, 12.0)[:, 0].T
    for m in upstream.modules():
        if isinstance(m, SConv1d):
            m.pad_mode = "constant"
    zero_latent = upstream.encoder(torch.from_numpy(x)[None, None])[0]
    zero_codes = upstream.quantizer.encode(zero_latent[None], 75, 12.0)[:, 0].T

enc = ref.StreamingEncoder(t, 16)
with torch.no_grad():
    latents = torch.stack([enc.embedding(x[i * 320:(i + 1) * 320]) for i in range(frames)], 1)
enc.reset()
codes = np.array([enc.quantize(enc.embedding(x[i * 320:(i + 1) * 320])) for i in range(frames)])

err = (latents - zero_latent).abs().max().item()
scale = zero_latent.abs().max().item()
print(f"1. encoder latent vs upstream (zero padding): max |diff| {err:.2e} (latent max {scale:.1f})")
print(f"   indices equal: {100 * (codes == zero_codes.numpy()).mean():.3f}% of {codes.size}")
diff_frames = np.nonzero((codes != reflect_codes.numpy()).any(axis=1))[0]
print(f"2. vs upstream as released (reflect): {100 * (codes == reflect_codes.numpy()).mean():.2f}% "
      f"indices equal; frames differing: {diff_frames[:20].tolist()}"
      f"{' ...' if len(diff_frames) > 20 else ''}")

vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").eval()
for n, bid in ((2, 0), (4, 1), (8, 2)):
    c = codes[:, :n]
    with torch.no_grad():
        feats = vocos.codes_to_features(torch.from_numpy(c.T.copy()).long())
        upstream_audio = vocos.decode(feats, bandwidth_id=torch.tensor([bid]))[0].numpy()
        batch = ref.decode_batch(t, c.tolist(), [bid] * frames)
    print(f"3. {n} codebooks: batch reference vs vocos max |diff| "
          f"{np.abs(batch - upstream_audio).max():.2e} (peak {np.abs(upstream_audio).max():.2f})")

for lookahead in (0, 3):
    c = codes[:, :4]
    with torch.no_grad():
        batch = ref.decode_window_batch(t, c.tolist(), [1] * frames, lookahead)
        dec = ref.StreamingDecoder(t, lookahead)
        stream = np.concatenate([dec.decode(ref.pack(row)) for row in c] + [dec.flush()])
    aligned = stream[dec.delay:]
    print(f"4. streaming + flush vs whole-signal, lookahead {lookahead}: delay {dec.delay}, "
          f"lengths {len(aligned)}/{len(batch)}, max |diff| {np.abs(aligned - batch).max():.2e}, "
          f"pre-roll max {np.abs(stream[:dec.delay]).max()}")
