#include <numeric>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <benchmark/benchmark.h>

#include "hardware_info.hpp"
#include "sgdlib/optimizer.hpp"

class LBFGS_Benchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        num_samples = state.range(0);
        num_features = state.range(1);
        search_policy_idx = state.range(2);  // 0 for Backtracking, 1 for Bracketing

        // Set search policy based on index
        if (search_policy_idx == 0) {
            search_policy = "BacktrackingLineSearch";
        } else {
            search_policy = "BracketingLineSearch";
        }

        // init DatasetManager
        data_manager = &sgdlib::DatasetManager<double, int>::get_instance(num_features);
        data_manager->set_max_samples(num_samples);
        data_manager->load(test_file);

        // init weight (no bias for LBFGS)
        w0.resize(num_features, 1.0);
    }

    void TearDown(const benchmark::State&) override {
        data_manager->clear();
    }

    void WriteOptimizerResults(benchmark::State& state, const sgdlib::LBFGS& optimizer) {
        std::filesystem::create_directories(log_folder_name);

        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        std::string policy_name = (search_policy_idx == 0) ? "backtracking" : "bracketing";
        std::string filename = log_folder_name + "/lbfgs_" + policy_name + "_result_" +
                                std::to_string(state.range(0)) + "x" +
                                std::to_string(state.range(1)) + "_" +
                                std::to_string(timestamp) + "_" +
                                std::to_string(state.iterations()) + ".log";

        // get result weights
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
        out << "Algorithm: L-BFGS\n";
        out << "Line Search: " << search_policy << "\n";
        out << "Samples size: " << state.range(0) << ", features size: "
            << state.range(1) << " features\n";

        out << "Weights (" << weights.size() << " dimensions):\n[";
        for (std::size_t i = 0; i < weights.size(); ++i) {
            if (i > 0) out << ", ";
            if (i % 5 == 0 && i != 0) out << "\n ";
            out << weights[i];
        }
        out << "]\n\n";

        // add stats info
        auto [min_it, max_it] = std::minmax_element(weights.begin(), weights.end());
        out << "Statistics:\n";
        out << "  Min weight:    " << *min_it << "\n";
        out << "  Max weight:    " << *max_it << "\n";
        out << "  Average weight: " << std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size() << "\n";

        out << "\n=== Benchmark Parameters ===\n";
        out << "Samples: " << state.range(0) << "\n";
        out << "Features: " << state.range(1) << "\n";
        out << "Search Policy: " << search_policy << "\n";
        out << "Iterations: " << state.iterations() << "\n";
    }

protected:
    sgdlib::DatasetManager<double, int>* data_manager;
    std::vector<double> w0;
    std::size_t num_samples;
    std::size_t num_features;
    int search_policy_idx;
    std::string search_policy;
    const std::string test_file = "./dataset/SUSY_processed";
    const std::string log_folder_name = "benchmark_results";
};

BENCHMARK_DEFINE_F(LBFGS_Benchmark, Optimize)(benchmark::State& state) {
    // get data
    const std::vector<double>& X = data_manager->get_features();
    const std::vector<int>& y = data_manager->get_labels();

    sgdlib::ArrayDatasetType dataset(X, y, num_samples, num_features);

    // L-BFGS parameters (from test_lbfgs.cpp)
    std::string loss = "LogLoss";
    double delta = 1e-6;
    double tol = 1e-5;
    std::size_t max_iters = 0;  // 0 means no limit
    std::size_t mem_size = 8;   // memory size for L-BFGS
    std::size_t past = 3;       // for convergence check
    bool verbose = false;

    // Step size search parameters (from test_lbfgs.cpp)
    std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params =
        std::make_shared<sgdlib::StepSizeSearchParamType>(
            sgdlib::detail::DEFAULT_STEPSIZE_SEARCH_PARAMS
        );
    stepsize_search_params->max_searches = 40;
    stepsize_search_params->max_iters = 20;

    // performance test loop
    for (auto _ : state) {
        sgdlib::LBFGS optimizer(
            w0,
            loss,
            search_policy,
            delta,
            tol,
            max_iters,
            mem_size,
            past,
            stepsize_search_params,
            verbose
        );

        optimizer.optimize(dataset);

        auto weights_copy = optimizer.get_weights();
        benchmark::DoNotOptimize(weights_copy);
        benchmark::ClobberMemory();
    }

    // Final run for reporting
    sgdlib::LBFGS final_optimizer(
        w0,
        loss,
        search_policy,
        delta,
        tol,
        max_iters,
        mem_size,
        past,
        stepsize_search_params,
        verbose
    );
    final_optimizer.optimize(dataset);
    WriteOptimizerResults(state, final_optimizer);

    // set stats info
    state.SetItemsProcessed(state.iterations() * num_samples);
    state.SetComplexityN(num_samples * num_features);
}

// L-BFGS with BacktrackingLineSearch
BENCHMARK_REGISTER_F(LBFGS_Benchmark, Optimize)
    ->Args({1000, 18, 0})      // 1k samples, Backtracking
    ->Args({100000, 18, 0})    // 10w samples, Backtracking
    ->Args({1000000, 18, 0})   // 100w samples, Backtracking
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

// L-BFGS with BracketingLineSearch
BENCHMARK_REGISTER_F(LBFGS_Benchmark, Optimize)
    ->Args({1000, 18, 1})      // 1k samples, Bracketing
    ->Args({100000, 18, 1})    // 10w samples, Bracketing
    ->Args({1000000, 18, 1})   // 100w samples, Bracketing
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

BENCHMARK_MAIN();
