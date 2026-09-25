#include "vocoscodec/model_file.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <stdexcept>

namespace vocoscodec {
namespace {

constexpr char kMagic[8] = {'V', 'O', 'C', 'O', 'S', 'C', 'M', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kAlign = 64;

bool LittleEndian() {
  const std::uint16_t probe = 1;
  std::uint8_t first;
  std::memcpy(&first, &probe, 1);
  return first == 1;
}

class Reader {
 public:
  Reader(const std::uint8_t* data, std::size_t size, const std::string& path)
      : data_(data), size_(size), path_(path) {}

  template <typename T>
  T Read() {
    Need(sizeof(T));
    T value;
    std::memcpy(&value, data_ + position_, sizeof(T));
    position_ += sizeof(T);
    return value;
  }

  std::string ReadString(std::size_t length) {
    Need(length);
    std::string value(reinterpret_cast<const char*>(data_ + position_), length);
    position_ += length;
    return value;
  }

 private:
  void Need(std::size_t n) {
    if (position_ + n > size_) {
      throw std::runtime_error(path_ + ": truncated model file");
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t position_ = 0;
  std::string path_;
};

std::string ShapeString(const std::vector<std::uint32_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) out << (i ? ", " : "") << shape[i];
  out << "]";
  return out.str();
}

}  // namespace

void ModelFile::AlignedFree::operator()(std::uint8_t* p) const {
  ::operator delete(p, std::align_val_t(kAlign));
}

ModelFile::ModelFile(const std::filesystem::path& path) : path_(path) {
  const std::string name = path.string();
  if (!LittleEndian()) {
    throw std::runtime_error("model files need a little-endian host");
  }
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error(name + ": cannot open model file");
  const std::streamoff size = in.tellg();
  if (size <= 0) throw std::runtime_error(name + ": empty model file");
  bytes_ = static_cast<std::size_t>(size);
  buffer_.reset(static_cast<std::uint8_t*>(
      ::operator new(bytes_, std::align_val_t(kAlign))));
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(buffer_.get()), size)) {
    throw std::runtime_error(name + ": cannot read model file");
  }

  Reader reader(buffer_.get(), bytes_, name);
  if (reader.ReadString(8) != std::string(kMagic, 8)) {
    throw std::runtime_error(name + ": not a vocoscodec model file");
  }
  const auto version = reader.Read<std::uint32_t>();
  if (version != kVersion) {
    throw std::runtime_error(name + ": model format version " + std::to_string(version) +
                             ", expected " + std::to_string(kVersion));
  }
  const auto count = reader.Read<std::uint32_t>();
  const auto meta_bytes = reader.Read<std::uint32_t>();
  reader.Read<std::uint32_t>();

  std::istringstream meta(reader.ReadString(meta_bytes));
  for (std::string line; std::getline(meta, line);) {
    const auto equals = line.find('=');
    if (equals != std::string::npos) meta_[line.substr(0, equals)] = line.substr(equals + 1);
  }

  for (std::uint32_t i = 0; i < count; ++i) {
    const auto length = reader.Read<std::uint16_t>();
    const std::string tensor_name = reader.ReadString(length);
    const auto dtype = reader.Read<std::uint8_t>();
    const auto rank = reader.Read<std::uint8_t>();
    Tensor tensor;
    tensor.size = 1;
    for (std::uint8_t d = 0; d < rank; ++d) {
      tensor.shape.push_back(reader.Read<std::uint32_t>());
      tensor.size *= tensor.shape.back();
    }
    const auto offset = reader.Read<std::uint64_t>();
    if (dtype != 0) {
      throw std::runtime_error(name + ": tensor " + tensor_name + " is not float32");
    }
    if (offset % alignof(float) != 0 || offset > bytes_ ||
        tensor.size > (bytes_ - offset) / sizeof(float)) {
      throw std::runtime_error(name + ": tensor " + tensor_name + " lies outside the file");
    }
    tensor.data = reinterpret_cast<float*>(buffer_.get() + offset);
    tensors_[tensor_name] = std::move(tensor);
  }
}

Tensor& ModelFile::Get(const std::string& name,
                       std::initializer_list<std::uint32_t> shape) {
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    throw std::runtime_error(path_.string() + ": missing tensor " + name);
  }
  if (!std::equal(shape.begin(), shape.end(), found->second.shape.begin(),
                  found->second.shape.end())) {
    throw std::runtime_error(path_.string() + ": tensor " + name + " has shape " +
                             ShapeString(found->second.shape) + ", expected " +
                             ShapeString(std::vector<std::uint32_t>(shape)));
  }
  return found->second;
}

bool ModelFile::Has(const std::string& name) const { return tensors_.count(name) != 0; }

std::string ModelFile::Meta(const std::string& key) const {
  const auto found = meta_.find(key);
  return found == meta_.end() ? std::string() : found->second;
}

}  // namespace vocoscodec
