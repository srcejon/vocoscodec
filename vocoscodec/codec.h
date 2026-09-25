// vocoscodec: the EnCodec 24 kHz encoder with the Vocos decoder as a speech
// codec. 24 kHz mono 16-bit audio in frames of 320 samples (13.33 ms); a
// packet per frame of 1..16 codebook indices of 10 bits (750 bit/s each).

#ifndef VOCOSCODEC_CODEC_H_
#define VOCOSCODEC_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vocoscodec {

class Codebooks;
class Concealer;
class EncoderModel;
class StreamingDecoder;
class StreamingEncoder;

inline constexpr int kCodecSampleRate = 24000;
inline constexpr int kCodecFrameSamples = 320;
inline constexpr int kCodecMaxCodebooks = 16;
inline constexpr int kCodecMaxPacketBytes = 20;
inline constexpr int kDefaultLookahead = 3;

// The model files a model directory must hold.
extern const char* const kModelFiles[3];
bool HasModels(const std::filesystem::path& model_directory);

class Encoder {
 public:
  // codebooks: 1..16 (0.75 to 12 kbit/s).
  static std::unique_ptr<Encoder> Create(const std::filesystem::path& model_directory,
                                         int codebooks, std::string* error);
  ~Encoder();

  // 320 samples -> one packet of PacketBytes(codebooks()) bytes.
  bool Encode(const std::int16_t* audio, std::size_t sample_count,
              std::vector<std::uint8_t>* packet, std::string* error);
  // The same from float samples in [-1, 1).
  bool Encode(const float* audio, std::size_t sample_count, std::vector<std::uint8_t>* packet,
              std::string* error);
  // Start of a transmission (key down): forget the past audio.
  void Reset();
  int codebooks() const { return codebooks_; }

 private:
  Encoder(std::unique_ptr<StreamingEncoder> encoder, int codebooks);

  std::unique_ptr<StreamingEncoder> encoder_;
  int codebooks_;
  std::vector<int> indices_;
  std::vector<float> samples_;
};

struct DecoderOptions {
  // Frames Vocos may look ahead (0..27). Each adds 320 samples of delay.
  int lookahead = kDefaultLookahead;
  // What long losses fade to: comfort noise, or silence.
  bool comfort_noise = true;
  // Tuning, normally left alone. Consecutive losses rendered from repeated
  // features, then frames of crossfade to comfort noise or silence.
  int conceal_repeat_frames = 3;
  int conceal_fade_frames = 3;
  // A fixed Vocos bandwidth embedding 0..3, or -1 for BandwidthId().
  int bandwidth_id = -1;
};

class Decoder {
 public:
  static std::unique_ptr<Decoder> Create(const std::filesystem::path& model_directory,
                                         const DecoderOptions& options, std::string* error);
  ~Decoder();

  // One packet (its size gives its codebook count) -> 320 samples appended.
  bool Decode(const std::uint8_t* packet, std::size_t packet_size,
              std::vector<std::int16_t>* audio, std::string* error);
  // Several consecutive packets -> 320 samples each, appended. A packet of
  // size 0 was lost and is concealed in its place. Frames early in a group
  // are decoded with the rest of the group as lookahead, which is better
  // than the minimum and cheaper per frame than one packet per call.
  bool Decode(const std::uint8_t* const* packets, const std::size_t* sizes, int count,
              std::vector<std::int16_t>* audio, std::string* error);
  bool Decode(const std::uint8_t* const* packets, const std::size_t* sizes, int count,
              std::vector<float>* audio, std::string* error);
  // Conceals one missing packet: 320 samples, advancing time by one frame.
  bool DecodeLostFrame(std::vector<std::int16_t>* audio, std::string* error);
  // End of a transmission: the last delay_samples() samples, then resets.
  bool Flush(std::vector<std::int16_t>* audio, std::string* error);
  bool Flush(std::vector<float>* audio, std::string* error);
  // True once concealment has faded fully to comfort noise (or silence).
  bool is_comfort_noise() const;
  // Start of a new transmission: forget the past.
  void Reset();
  // Output sample n corresponds to input sample n - delay_samples().
  int delay_samples() const;
  int lookahead() const;
  // ConvNeXt block evaluations so far (a cost measure for benchmarks).
  long long block_evaluations() const;

 private:
  Decoder(std::unique_ptr<StreamingDecoder> decoder, std::shared_ptr<const Codebooks> books,
          const DecoderOptions& options);
  bool DecodeFloat(const std::uint8_t* const* packets, const std::size_t* sizes, int count,
                   std::string* error);

  std::unique_ptr<StreamingDecoder> decoder_;
  std::shared_ptr<const Codebooks> books_;
  std::unique_ptr<Concealer> concealer_;
  std::vector<float> features_, noise_weights_, output_;
  std::vector<int> ids_;
  std::vector<std::uint8_t> concealed_;
  int bandwidth_id_;
};

// Vocos bandwidth embedding (0..3 for 1.5/3/6/12 kbit/s) used for a packet
// of `codebooks`.
int BandwidthId(int codebooks);

}  // namespace vocoscodec

#endif  // VOCOSCODEC_CODEC_H_
