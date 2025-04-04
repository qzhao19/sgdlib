#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <benchmark/benchmark.h>

#include "hardware_info.hpp"
#include "sgdlib/optimizer.hpp"

class SGD_Benchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        num_samples = state.range(0);
        num_features = state.range(1);
        
        // init DatasetManager
        data_manager = &sgdlib::DatasetManager<double, long>::get_instance(num_features);
        data_manager->set_max_samples(num_samples);
        data_manager->load(test_file);

        // init weight and bias
        w0.resize(num_features, 1.0);  
        b0 = 0.0;
    }

    void TearDown(const benchmark::State&) override {
        data_manager->clear();
    }

    void WriteOptimizerResults(benchmark::State& state, const sgdlib::SGD& optimizer) {
        std::filesystem::create_directory(log_folder_name);
    
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        std::string filename = "benchmark_results/sgd_result_" +
                                std::to_string(state.range(0)) + "x" +
                                std::to_string(state.range(1)) + "_" +
                                std::to_string(timestamp) + "_" +
                                std::to_string(state.iterations()) + ".log";
        
        // get result w, b
        double intercept = optimizer.get_intercept();
        const std::vector<double>& weights = optimizer.get_weights();
        
        // get hardware info
        auto hardware_info = HardwareInfo::get_sys_info();

        // write file
        std::ofstream out(filename);
        out << "=== System Hardware Info ===\n";
        out << "CPU: " << hardware_info["cpu_name"] << "\n";
        out << "Cores: " << hardware_info["cpu_cores"] << "\n";
        out << "Memory: " << hardware_info["memory_total"] 
            << " (Free: " << hardware_info["memory_free"] << ")\n";
        out << "Environment: " << hardware_info["wsl_version"] << "\n\n";

        out << std::scientific << std::setprecision(15);
        out << "=== Optimization Results ===\n";
        out << "Samples size: " << state.range(0) << ", features size: " 
            << state.range(1) << " features\n";
        
        out << "Weights (" << weights.size() << " dimensions):\n[";
        for (std::size_t i = 0; i < weights.size(); ++i) {
            if (i > 0) out << ", ";
            if (i % 5 == 0 && i != 0) out << "\n ";
            out << weights[i];
        }
        out << "]\n\n";
        out << "Intercept: " << intercept << "\n\n";
        
        // add stats info
        auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
        out << "Statistics:\n";
        out << "  Min weight:    " << *min_it << "\n";
        out << "  Max weight:    " << *max_it << "\n";
        out << "  Average weight: " << std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size() << "\n";
        
        out << "\n=== Benchmark Parameters ===\n";
        out << "Samples: " << state.range(0) << "\n";
        out << "Features: " << state.range(1) << "\n";
        out << "Iterations: " << state.iterations() << "\n";
        // out << "Time: " << state.iterations() * state.iterations_time().count() << " ns\n";
    }

protected:
    sgdlib::DatasetManager<double, long>* data_manager;
    std::vector<double> w0;
    double b0;
    std::size_t num_samples;
    std::size_t num_features;
    const std::string test_file = "./dataset/SUSY_processed";
    const std::string log_folder_name = "benchmark_results";
};

BENCHMARK_DEFINE_F(SGD_Benchmark, Optimize)(benchmark::State& state) {
    // 创建SGD优化器
    sgdlib::SGD optimizer(
        w0, b0,                    
        "LogLoss", 
        "Invscaling",
        0.0,
        0.01,
        1e-4,
        0.2,
        100,
        1,
        5,
        -1,
        true,
        false
    );

    // get data
    const std::vector<double>& X = data_manager->get_features();
    const std::vector<long>& y = data_manager->get_labels();

    // performance test loop
    for (auto _ : state) {
        optimizer.optimize(X, y);
    }

    // report results
    WriteOptimizerResults(state, optimizer);

    // set stats info
    state.SetItemsProcessed(state.iterations() * num_samples);
    state.SetComplexityN(num_samples * num_features);
}

// different data size
BENCHMARK_REGISTER_F(SGD_Benchmark, Optimize)
    ->Args({1000, 18})     // 1k samples
    // ->Args({5000, 18})     // 5k samples
    // ->Args({10000, 18})     // 10k samples
    // ->Args({50000, 18})     // 50k samples
    ->Args({100000, 18})    // 10w samples
    // ->Args({500000, 18})    // 50w samples
    // ->Args({1000000, 18})   // 100w samples
    ->Args({2000000, 18})   // 200w samples
    // ->Args({3000000, 18})   // 300w samples
    // ->Args({4000000, 18})   // 400w samples
    // ->Args({4500000, 18})   // 450w samples
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

BENCHMARK_MAIN();

