// Reader for vocoscodec model files, format version 1 (see README.md,
// "Model files", and tools/modelfile.py).

#ifndef VOCOSCODEC_MODEL_FILE_H_
#define VOCOSCODEC_MODEL_FILE_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vocoscodec {

struct Tensor {
  std::vector<std::uint32_t> shape;
  float* data = nullptr;
  std::size_t size = 0;
};

class ModelFile {
 public:
  // Reads the whole file. Throws std::runtime_error with the reason on failure.
  explicit ModelFile(const std::filesystem::path& path);

  // The tensor `name` with exactly `shape`; throws if missing or mismatched.
  Tensor& Get(const std::string& name, std::initializer_list<std::uint32_t> shape);
  bool Has(const std::string& name) const;
  // Metadata value, or empty if absent.
  std::string Meta(const std::string& key) const;
  const std::filesystem::path& path() const { return path_; }
  std::size_t bytes() const { return bytes_; }

 private:
  struct AlignedFree {
    void operator()(std::uint8_t* p) const;
  };

  std::filesystem::path path_;
  std::unique_ptr<std::uint8_t, AlignedFree> buffer_;
  std::size_t bytes_ = 0;
  std::map<std::string, std::string> meta_;
  std::map<std::string, Tensor> tensors_;
};

}  // namespace vocoscodec

#endif  // VOCOSCODEC_MODEL_FILE_H_
