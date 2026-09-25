"""Arithmetic and weight traffic per frame, for estimating other CPUs.

Counts multiply-accumulates (MAC) and the weight bytes read per frame by the
encoder and by the decoder (with a given ConvNeXt block count per frame, as
vocoscodec_golden_test and vocoscodec_benchmark report), with the half
precision layers of the default build.
"""
FRAME = 320


def encoder():
    macs, weight_bytes = 0, 0
    steps, channels = FRAME, 32
    macs += steps * 32 * 7
    weight_bytes += 32 * 7 * 4
    for stride in (2, 4, 5, 8):
        c = channels
        layer = [(c // 2, c * 3), (c, c // 2), (c, c)]  # conv1 k3, conv2 k1, skip k1
        for out, depth in layer:
            macs += steps * out * depth
            weight_bytes += out * depth * 4
        out, depth = 2 * c, c * 2 * stride
        steps //= stride
        macs += steps * out * depth
        weight_bytes += out * depth * (2 if stride == 8 else 4)  # down3 in half
        channels *= 2
    lstm = 2 * 2 * (4 * 512 * 512)
    macs += lstm
    weight_bytes += lstm * 2
    macs += 128 * 512 * 7
    weight_bytes += 128 * 512 * 7 * 2
    return macs, weight_bytes


def decoder(blocks_per_frame: float, frames_per_call: int):
    block = 384 * 1152 * 2 + 384 * 7
    embed, head = 384 * 128 * 7, 1282 * 384
    # The cone recomputes about half as many embeddings as blocks per layer.
    macs = blocks_per_frame * block + (blocks_per_frame / 8 + 1) * embed + head
    weights = (8 * 384 * 1152 * 2 + 384 * 128 * 7 + 1282 * 384) * 2  # half precision
    return macs, weights / frames_per_call


if __name__ == "__main__":
    m, b = encoder()
    print(f"encoder: {m / 1e6:.1f} M MAC/frame, {b / 1e6:.1f} MB weights read/frame")
    for blocks, group in ((75.7, 1), (17.6, 9), (13.0, 18)):
        m, b = decoder(blocks, group)
        print(f"decoder, {group:2d} per call ({blocks} blocks/frame): {m / 1e6:.1f} M MAC/frame, "
              f"{b / 1e6:.1f} MB weights/frame")
