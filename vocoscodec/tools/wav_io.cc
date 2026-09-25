#include "vocoscodec/tools/wav_io.h"

#include <cstring>
#include <fstream>
#include <iterator>

namespace vocoscodec::tools {
namespace {

std::uint32_t U32(const std::uint8_t* p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint16_t U16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

void Put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
void Put16(std::vector<std::uint8_t>& out, std::uint16_t v) {
  out.push_back(static_cast<std::uint8_t>(v));
  out.push_back(static_cast<std::uint8_t>(v >> 8));
}

bool ReadFile(const std::filesystem::path& path, std::vector<std::uint8_t>* data,
              std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path.string();
    return false;
  }
  data->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

bool WriteFile(const std::filesystem::path& path, const std::vector<std::uint8_t>& data,
               std::string* error) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!out) {
    *error = "cannot write " + path.string();
    return false;
  }
  return true;
}

}  // namespace

bool ReadMonoWav(const std::filesystem::path& path, std::vector<std::int16_t>* samples,
                 int* sample_rate, std::string* error) {
  std::vector<std::uint8_t> data;
  if (!ReadFile(path, &data, error)) return false;
  if (data.size() < 12 || std::memcmp(data.data(), "RIFF", 4) || std::memcmp(data.data() + 8, "WAVE", 4)) {
    *error = path.string() + ": not a WAV file";
    return false;
  }
  bool have_format = false;
  for (std::size_t pos = 12; pos + 8 <= data.size();) {
    const std::uint32_t size = U32(&data[pos + 4]);
    const std::uint8_t* body = &data[pos + 8];
    if (pos + 8 + size > data.size()) break;
    if (!std::memcmp(&data[pos], "fmt ", 4) && size >= 16) {
      if (U16(body) != 1 || U16(body + 2) != 1 || U16(body + 14) != 16) {
        *error = path.string() + ": expected mono 16-bit PCM";
        return false;
      }
      *sample_rate = static_cast<int>(U32(body + 4));
      have_format = true;
    } else if (!std::memcmp(&data[pos], "data", 4) && have_format) {
      samples->resize(size / 2);
      for (std::size_t i = 0; i < samples->size(); ++i)
        (*samples)[i] = static_cast<std::int16_t>(U16(body + 2 * i));
      return true;
    }
    pos += 8 + size + (size & 1);
  }
  *error = path.string() + ": no PCM data";
  return false;
}

bool WriteMonoWav(const std::filesystem::path& path, const std::vector<std::int16_t>& samples,
                  int sample_rate, std::string* error) {
  std::vector<std::uint8_t> out;
  const auto bytes = static_cast<std::uint32_t>(samples.size() * 2);
  out.insert(out.end(), {'R', 'I', 'F', 'F'});
  Put32(out, 36 + bytes);
  out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  Put32(out, 16);
  Put16(out, 1);
  Put16(out, 1);
  Put32(out, static_cast<std::uint32_t>(sample_rate));
  Put32(out, static_cast<std::uint32_t>(sample_rate) * 2);
  Put16(out, 2);
  Put16(out, 16);
  out.insert(out.end(), {'d', 'a', 't', 'a'});
  Put32(out, bytes);
  for (std::int16_t s : samples) Put16(out, static_cast<std::uint16_t>(s));
  return WriteFile(path, out, error);
}

bool ReadPackets(const std::filesystem::path& path,
                 std::vector<std::vector<std::uint8_t>>* packets, std::string* error) {
  std::vector<std::uint8_t> data;
  if (!ReadFile(path, &data, error)) return false;
  for (std::size_t pos = 0; pos < data.size();) {
    const std::size_t size = data[pos++];
    if (pos + size > data.size()) {
      *error = path.string() + ": truncated packet file";
      return false;
    }
    packets->emplace_back(data.begin() + pos, data.begin() + pos + size);
    pos += size;
  }
  return true;
}

bool WritePackets(const std::filesystem::path& path,
                  const std::vector<std::vector<std::uint8_t>>& packets, std::string* error) {
  std::vector<std::uint8_t> out;
  for (const auto& packet : packets) {
    out.push_back(static_cast<std::uint8_t>(packet.size()));
    out.insert(out.end(), packet.begin(), packet.end());
  }
  return WriteFile(path, out, error);
}

}  // namespace vocoscodec::tools
