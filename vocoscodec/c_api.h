/* vocoscodec C ABI: the EnCodec 24 kHz encoder with the Vocos decoder.
 *
 * Audio is 24 kHz mono 16-bit PCM in frames of 320 samples (13.33 ms). A
 * packet carries one frame as 1..16 codebook indices of 10 bits (750 bit/s
 * per codebook); see README.md "Packets" for the normative layout.
 *
 * Every function is safe to call with a null handle. Functions returning a
 * count return 0 on failure and set the calling thread's last error. */

#ifndef VOCOSCODEC_C_API_H_
#define VOCOSCODEC_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(VOCOSCODEC_RUNTIME_BUILD)
#define VOCOSCODEC_API __declspec(dllexport)
#else
#define VOCOSCODEC_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define VOCOSCODEC_API __attribute__((visibility("default")))
#else
#define VOCOSCODEC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VOCOSCODEC_ABI_VERSION 1
#define VOCOSCODEC_SAMPLE_RATE 24000
#define VOCOSCODEC_SAMPLES_PER_FRAME 320
#define VOCOSCODEC_MAX_CODEBOOKS 16
#define VOCOSCODEC_BITS_PER_CODEBOOK 10
#define VOCOSCODEC_MAX_PACKET_BYTES 20
#define VOCOSCODEC_DEFAULT_LOOKAHEAD 3
#define VOCOSCODEC_MAX_LOOKAHEAD 27

/* vocoscodec_decoder_create flags */
#define VOCOSCODEC_DECODER_SILENCE 1 /* long losses fade to silence, not comfort noise */

typedef struct vocoscodec_encoder vocoscodec_encoder;
typedef struct vocoscodec_decoder vocoscodec_decoder;

VOCOSCODEC_API int vocoscodec_abi_version(void);
VOCOSCODEC_API const char* vocoscodec_version(void);
/* The model directory the library was built with (may not exist here). */
VOCOSCODEC_API const char* vocoscodec_built_model_directory(void);
/* 1 when the directory holds the three model files. */
VOCOSCODEC_API int vocoscodec_has_models(const char* model_directory);
/* Why the calling thread's last call failed (empty after a success). */
VOCOSCODEC_API const char* vocoscodec_last_error(void);

/* Bytes in a packet of `codebooks` (1..16), 0 otherwise. */
VOCOSCODEC_API int vocoscodec_packet_bytes(int codebooks);
/* Codebooks in a packet of `packet_bytes`, 0 if no count gives that size. */
VOCOSCODEC_API int vocoscodec_packet_codebooks(int packet_bytes);
/* Truncates a packet in place to fewer codebooks (as a relay does); returns
 * the new size, 0 if the packet or count is invalid. */
VOCOSCODEC_API int vocoscodec_packet_truncate(uint8_t* packet, int packet_bytes,
                                              int codebooks);

/* codebooks: 1..16 (0.75 to 12 kbit/s). */
VOCOSCODEC_API vocoscodec_encoder* vocoscodec_encoder_create(const char* model_directory,
                                                             int codebooks);
/* 320 samples -> one packet; returns its size in bytes. */
VOCOSCODEC_API int vocoscodec_encode(vocoscodec_encoder* encoder, const int16_t* pcm,
                                     uint8_t* packet, int packet_capacity);
/* Start of a transmission (key down). */
VOCOSCODEC_API void vocoscodec_encoder_reset(vocoscodec_encoder* encoder);
VOCOSCODEC_API void vocoscodec_encoder_destroy(vocoscodec_encoder* encoder);

/* lookahead: frames Vocos may look ahead, 0..27, or -1 for the default (3).
 * flags: 0 or VOCOSCODEC_DECODER_SILENCE. */
VOCOSCODEC_API vocoscodec_decoder* vocoscodec_decoder_create(const char* model_directory,
                                                             int lookahead, int flags);
/* Output sample n corresponds to input sample n - delay (480 + 320 lookahead). */
VOCOSCODEC_API int vocoscodec_decoder_delay_samples(vocoscodec_decoder* decoder);
/* One packet (its size gives its codebook count) -> 320 samples; returns 320. */
VOCOSCODEC_API int vocoscodec_decode(vocoscodec_decoder* decoder, const uint8_t* packet,
                                     int packet_bytes, int16_t* pcm, int pcm_capacity);
/* `count` consecutive packets -> 320 samples each; returns 320 * count. A
 * packet with packet_bytes[i] == 0 was lost and is concealed in its place.
 * Frames early in the group use the rest of it as lookahead: better quality
 * than one packet per call, and much less computation per frame. */
VOCOSCODEC_API int vocoscodec_decode_packets(vocoscodec_decoder* decoder,
                                             const uint8_t* const* packets,
                                             const int* packet_bytes, int count,
                                             int16_t* pcm, int pcm_capacity);
/* Conceals one missing packet: 320 samples, advancing time by one frame. */
VOCOSCODEC_API int vocoscodec_decode_lost(vocoscodec_decoder* decoder, int16_t* pcm,
                                          int pcm_capacity);
/* End of a transmission: the last delay samples of audio, then resets.
 * Returns the delay. */
VOCOSCODEC_API int vocoscodec_decoder_flush(vocoscodec_decoder* decoder, int16_t* pcm,
                                            int pcm_capacity);
/* 1 once concealment has faded fully to comfort noise (or silence). */
VOCOSCODEC_API int vocoscodec_decoder_is_comfort_noise(vocoscodec_decoder* decoder);
/* Start of a new transmission. */
VOCOSCODEC_API void vocoscodec_decoder_reset(vocoscodec_decoder* decoder);
VOCOSCODEC_API void vocoscodec_decoder_destroy(vocoscodec_decoder* decoder);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VOCOSCODEC_C_API_H_ */
