#include "vocoscodec/concealment.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vocoscodec {

void Concealer::Received(const float* features, int bandwidth_id) {
  std::copy(features, features + kLatentDim, last_);
  last_id_ = bandwidth_id;
  lost_ = 0;
}

void Concealer::Lost(float* features, int* bandwidth_id, float* noise_weight) {
  ++lost_;
  std::copy(last_, last_ + kLatentDim, features);
  *bandwidth_id = last_id_;
  const int into_fade = lost_ - parameters_.repeat_frames;
  if (into_fade <= 0) {
    *noise_weight = 0.0f;
  } else if (parameters_.fade_frames <= 0 || into_fade >= parameters_.fade_frames) {
    *noise_weight = 1.0f;
  } else {
    *noise_weight = static_cast<float>(into_fade) / parameters_.fade_frames;
  }
}

void Concealer::Reset() {
  std::fill(last_, last_ + kLatentDim, 0.0f);
  last_id_ = 0;
  lost_ = 0;
}

void ComfortNoise::Reset() {
  magnitudes_.assign(static_cast<std::size_t>(kHistory) * kBins, 0.0f);
  energy_.assign(kHistory, 0.0f);
  floor_.assign(kBins, 0.0f);
  count_ = 0;
  next_ = 0;
  floor_stale_ = true;
  random_ = 0x6d2b79f5u;
}

void ComfortNoise::Observe(const float* magnitudes) {
  float* slot = magnitudes_.data() + static_cast<std::size_t>(next_) * kBins;
  std::copy(magnitudes, magnitudes + kBins, slot);
  float energy = 0.0f;
  for (int k = 0; k < kBins; ++k) energy += magnitudes[k] * magnitudes[k];
  energy_[next_] = energy;
  next_ = (next_ + 1) % kHistory;
  count_ = std::min(count_ + 1, kHistory);
  floor_stale_ = true;
}

// The mean magnitude spectrum of the quietest fifth of the last second.
void ComfortNoise::UpdateFloor() {
  std::fill(floor_.begin(), floor_.end(), 0.0f);
  if (count_ > 0) {
    std::vector<int> order(count_);
    std::iota(order.begin(), order.end(), 0);
    const int quietest = std::max(1, count_ * kQuietest / kHistory);
    std::partial_sort(order.begin(), order.begin() + quietest, order.end(),
                      [this](int a, int b) {
                        return energy_[a] < energy_[b] || (energy_[a] == energy_[b] && a < b);
                      });
    for (int i = 0; i < quietest; ++i) {
      const float* m = magnitudes_.data() + static_cast<std::size_t>(order[i]) * kBins;
      for (int k = 0; k < kBins; ++k) floor_[k] += m[k] / quietest;
    }
  }
  floor_stale_ = false;
}

void ComfortNoise::Mix(std::complex<float>* spectrum, float weight) {
  if (weight <= 0.0f) return;
  if (enabled_ && floor_stale_) UpdateFloor();
  const float keep = 1.0f - weight;
  constexpr float kTwoPi = 6.28318530717958648f;
  for (int k = 0; k < kBins; ++k) {
    std::complex<float> noise(0.0f, 0.0f);
    if (enabled_) {
      // xorshift32; the top 24 bits give the phase.
      random_ ^= random_ << 13;
      random_ ^= random_ >> 17;
      random_ ^= random_ << 5;
      const float phase = static_cast<float>(random_ >> 8) * (kTwoPi / 16777216.0f);
      noise = std::polar(floor_[k], phase);
    }
    spectrum[k] = keep * spectrum[k] + weight * noise;
  }
}

}  // namespace vocoscodec
