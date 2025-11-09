#include <numeric>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <benchmark/benchmark.h>

#include "hardware_info.hpp"
#include "sgdlib/optimizer.hpp"

class SAG_Benchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        num_samples = state.range(0);
        num_features = state.range(1);
        is_saga = state.range(2);  // 0 for SAG, 1 for SAGA

        // init DatasetManager
        data_manager = &sgdlib::DatasetManager<double, int>::get_instance(num_features);
        data_manager->set_max_samples(num_samples);
        data_manager->load(test_file);

        // init weight and bias
        w0.resize(num_features, 1.0);
        b0 = 0.0;
    }

    void TearDown(const benchmark::State&) override {
        data_manager->clear();
    }

    void WriteOptimizerResults(benchmark::State& state, const sgdlib::SAG& optimizer) {
        std::filesystem::create_directories(log_folder_name);

        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        std::string algo_name = is_saga ? "saga" : "sag";
        std::string filename = log_folder_name + "/" + algo_name + "_result_" +
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
        out << "Algorithm: " << (is_saga ? "SAGA" : "SAG") << "\n";
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
        out << "Is SAGA: " << (is_saga ? "true" : "false") << "\n";
        out << "Iterations: " << state.iterations() << "\n";
    }

protected:
    sgdlib::DatasetManager<double, int>* data_manager;
    std::vector<double> w0;
    double b0;
    std::size_t num_samples;
    std::size_t num_features;
    bool is_saga;
    const std::string test_file = "./dataset/SUSY_processed";
    const std::string log_folder_name = "benchmark_results";
};

BENCHMARK_DEFINE_F(SAG_Benchmark, Optimize)(benchmark::State& state) {
    // get data
    const std::vector<double>& X = data_manager->get_features();
    const std::vector<int>& y = data_manager->get_labels();

    sgdlib::ArrayDatasetType dataset(X, y, num_samples, num_features);

    // performance test loop
    for (auto _ : state) {
        sgdlib::SAG optimizer(
            w0, b0,
            "LogLoss",              // loss function
            "ExactLineSearch",      // search policy (from test_sag.cpp)
            0.0,                    // alpha (regularization)
            0.01,                   // eta0 (learning rate)
            0.005,                  // tol (tolerance, from test_sag.cpp)
            100,                    // max_iters
            1,                      // random_seed
            is_saga,                // is_saga flag
            true,                   // shuffle
            false                   // verbose
        );
        optimizer.optimize(dataset);

        auto weights_copy = optimizer.get_weights();
        benchmark::DoNotOptimize(weights_copy);
        benchmark::ClobberMemory();
    }

    sgdlib::SAG final_optimizer(
        w0, b0,
        "LogLoss",
        "ExactLineSearch",
        0.0,
        0.01,
        0.005,
        100,
        1,
        is_saga,
        true,
        false
    );
    final_optimizer.optimize(dataset);
    // report results
    WriteOptimizerResults(state, final_optimizer);

    // set stats info
    state.SetItemsProcessed(state.iterations() * num_samples);
    state.SetComplexityN(num_samples * num_features);
}

// SAG benchmark (is_saga = 0)
BENCHMARK_REGISTER_F(SAG_Benchmark, Optimize)
    ->Args({1000, 18, 0})     // 1k samples, SAG
    ->Args({100000, 18, 0})   // 10w samples, SAG
    ->Args({1000000, 18, 0})  // 100w samples, SAG
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

// SAGA benchmark (is_saga = 1)
BENCHMARK_REGISTER_F(SAG_Benchmark, Optimize)
    ->Args({1000, 18, 1})     // 1k samples, SAGA
    ->Args({100000, 18, 1})   // 10w samples, SAGA
    ->Args({1000000, 18, 1})  // 100w samples, SAGA
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

BENCHMARK_MAIN();
