#include <xrt/xrt_device.h>
#include <xrt/experimental/xrt_ip.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
  if (argc != 4) {
    std::cerr << "usage: xrt-register-server DEVICE XCLBIN KERNEL\n";
    return 2;
  }
  xrt::device device{static_cast<unsigned int>(std::stoul(argv[1], nullptr, 0))};
  auto uuid = device.load_xclbin(argv[2]);
  xrt::ip ip{device, uuid, argv[3]};
  std::cout << "READY" << std::endl;

  std::string operation;
  while (std::cin >> operation) {
    if (operation == "q") break;
    std::string offset_text;
    std::cin >> offset_text;
    auto offset = static_cast<std::uint32_t>(std::stoul(offset_text, nullptr, 0));
    if (operation == "r") {
      std::cout << "OK " << ip.read_register(offset) << std::endl;
    } else if (operation == "w") {
      std::string value_text;
      std::cin >> value_text;
      auto value = static_cast<std::uint32_t>(std::stoul(value_text, nullptr, 0));
      ip.write_register(offset, value);
      std::cout << "OK" << std::endl;
    } else {
      std::cout << "ERROR unknown-operation" << std::endl;
    }
  }
  return 0;
} catch (const std::exception& error) {
  std::cerr << "xrt-register-server: " << error.what() << '\n';
  return 1;
}
