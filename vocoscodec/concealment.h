// Lost packet concealment.
//
// A lost frame is given the features of the last frame received, so Vocos
// continues the sound, with real frames on its right once the lookahead
// has them. After `repeat_frames` consecutive losses, the frame's spectrum
// crossfades over `fade_frames` to comfort noise. Its magnitudes are the
// noise floor of the frames received in the last second, and its phases are
// random. With comfort noise off, the crossfade goes to silence. Mixing in
// the spectrum lets the inverse STFT's overlap-add smooth every transition.

#ifndef VOCOSCODEC_CONCEALMENT_H_
#define VOCOSCODEC_CONCEALMENT_H_

#include <complex>
#include <cstdint>
#include <vector>

#include "vocoscodec/encoder.h"

namespace vocoscodec {

struct ConcealmentParameters {
  int repeat_frames = 3;  // losses rendered from repeated features alone
  int fade_frames = 3;    // then the crossfade to comfort noise or silence
};

// Chooses the decoder's input for each frame, received or lost.
class Concealer {
 public:
  explicit Concealer(const ConcealmentParameters& parameters) : parameters_(parameters) {}

  void Received(const float* features, int bandwidth_id);
  // Features, bandwidth id and comfort noise weight (0..1) for a lost frame.
  void Lost(float* features, int* bandwidth_id, float* noise_weight);
  void Reset();
  int consecutive_lost() const { return lost_; }

 private:
  ConcealmentParameters parameters_;
  float last_[kLatentDim] = {};
  int last_id_ = 0;
  int lost_ = 0;
};

// Noise floor tracking and comfort noise spectra.
class ComfortNoise {
 public:
  static constexpr int kBins = 641;
  static constexpr int kHistory = 75;  // one second of frames
  static constexpr int kQuietest = 15;  // the floor is their mean magnitude

  explicit ComfortNoise(bool enabled) : enabled_(enabled) { Reset(); }

  // A frame decoded from a received packet: its magnitudes.
  void Observe(const float* magnitudes);
  // spectrum = (1 - weight) spectrum + weight * noise (silence when disabled).
  void Mix(std::complex<float>* spectrum, float weight);
  void Reset();

 private:
  void UpdateFloor();

  bool enabled_;
  std::vector<float> magnitudes_;  // [kHistory][kBins]
  std::vector<float> energy_;
  int count_ = 0, next_ = 0;
  bool floor_stale_ = true;
  std::vector<float> floor_;
  std::uint32_t random_ = 0x6d2b79f5u;
};

}  // namespace vocoscodec

#endif  // VOCOSCODEC_CONCEALMENT_H_
