// hardware_info.hpp
#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cpuid.h> // For CPU info (Linux/WSL)
#include <thread> // For CPU cores

class HardwareInfo {
public:
    HardwareInfo() = default;
    ~HardwareInfo() = default;
    
    static std::unordered_map<std::string, std::string> get_sys_info() {
        std::unordered_map<std::string, std::string> info;
        
        // CPU Info
        info["cpu_name"] = get_cpu_name();
        info["cpu_cores"] = std::to_string(std::thread::hardware_concurrency());
        
        // Memory Info
        auto meminfo = get_memory_info();
        info["memory_total"] = meminfo["MemTotal"];
        info["memory_free"] = meminfo["MemFree"];
        
        // WSL Specific (if applicable)
        info["wsl_version"] = get_wsl_version();
        
        return info;
    }

private:
    static std::string get_cpu_name() {
        char brand[0x40] = {0};
        unsigned int *brand_uint = reinterpret_cast<unsigned int*>(brand);
        
        __get_cpuid(0x80000002, brand_uint+0x00, brand_uint+0x01, brand_uint+0x02, brand_uint+0x03);
        __get_cpuid(0x80000003, brand_uint+0x04, brand_uint+0x05, brand_uint+0x06, brand_uint+0x07);
        __get_cpuid(0x80000004, brand_uint+0x08, brand_uint+0x09, brand_uint+0x0A, brand_uint+0x0B);
        
        return std::string(brand);
    }

    static std::unordered_map<std::string, std::string> get_memory_info() {
        std::unordered_map<std::string, std::string> meminfo;
        std::ifstream file("/proc/meminfo");
        std::string line;
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            if (!key.empty()) {
                key.pop_back(); // Remove ':'
                meminfo[key] = value + " " + unit;
            }
        }
        return meminfo;
    }

    static std::string get_wsl_version() {
        std::ifstream file("/proc/version");
        std::string version;
        std::getline(file, version);
        
        if (version.find("Microsoft") != std::string::npos) {
            return version.substr(version.find("Microsoft"));
        }
        return "Native Linux";
    }
};
