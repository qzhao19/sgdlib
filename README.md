# sgdlib
sgdlib: A header-only C++ Library for Optimization Algorithms

# Overview

In the field of machine learning, particularly in statistical machine learning, optimization algorithms lie at the heart of nearly every problem. Optimization algorithms are the backbone of machine learning systems, playing a pivotal role in determining their robustness， efficiency, and overall performance.

sgdlib is a lightweight and efficient C++ library designed to simplify the implementation and experimentation of various optimization algorithms, with a particular focus on Stochastic Gradient Descent (SGD) and its variants. The library provides a modular and extensible framework, enabling users to easily implement optimization algorithms, loss functions, learning rate decay strategies, and step size search methods.

With its straightforward API interfaces, the library makes it effortless for users to leverage a wide range of optimization algorithms. Moreover, it supports the integration of custom objective functions through a uniform interface naming convention. This design ensures flexibility and extensibility, while encapsulating the core internal algorithms. As a result, users can concentrate on their specific optimization tasks without being burdened by the complexities of the underlying implementation.

The library includes the following optimization algorithms:

- Stochastic Gradient Descent (SGD)
- Stochastic Average Gradient (SAG)
- Stochastic Variance Reduced Gradient (SVRG)
- Limited-memory BFGS (LBFGS)
- Stochastic Coordinate Descent (SCD)

## Key Features

- **Modular and Decoupled Design**:
  - sgdlib embraces a modular and decoupled design philosophy, where the optimization process is logically divided into independent components: main logic operations, loss function computations, learning rate decay strategies, and step size search methods. Each component is self-contained and interacts with others through a unified API interface. This design ensures that the core logic remains clean and maintainable, while allowing users to easily extend or replace individual components without affecting the overall system. By promoting modularity and reusability, this approach significantly enhances the flexibility and scalability of sgdlib.

- **Registration Mechanism**:
  - Flexible and extensible registration mechanism allows users to define custom loss functions, learning rate decay strategies and step size search policies without modifying the core framework. This key feature ensures that sgdlib remains adaptable to a wide range of optimization tasks while maintaining a clean and modular design.

- **Streamlined and Intuitive API Design**:
  - sgdlib features a streamlined and intuitive API design, inspired by the widely-used scikit-learn interface. This design ensures that the API is consistent, easy to understand, and quick to learn, enabling users to get started with minimal effort. Key methods such as `optimize()`, `get_coef()`, and optional `get_intercept()` are implemented to align with the design principles of popular machine learning libraries, making sgdlib a seamless choice for users transitioning from Python to C++ for optimization tasks.

- **Native C++ Implementation**:
  - sgdlib is built entirely using native C++, with no dependencies on external third-party libraries. This ensures maximum performance and portability, making it suitable for a wide range of environments, from embedded systems to high-performance computing clusters. By leveraging the full power of modern C++, sgdlib delivers efficient and reliable optimization capabilities without the overhead of additional dependencies.

## System Requirements

To build and use sgdlib, the following tools and libraries are required:

- **C++ Compiler**: A C++17-compliant compiler (e.g., GCC 9+, Clang 10+, or MSVC 2019+).
- **CMake**: Version 3.26.5 or higher.
- **Google Test (GTest)**: Version 1.14.0 or higher (for running unit tests).





