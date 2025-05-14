# OCaml-PJRT
To install this library, run the following command:

``` shell
opam pin add ocaml_bayes_dl https://github.com/MichelBartels/device-api.git
```

To make any use of this library, you also need a PJRT plugin for your backend. There are multiple options:

## XLA (recommended)
XLA supports various accelerators (CPU, CUDA, ROCm, SYCL). To build it, follow the [steps in the official documentation](https://openxla.org/xla/build_from_source). Replace `//xla/...` with `//xla/pjrt/c:pjrt_c_api_cpu_plugin.so` for the CPU backend or `//xla/pjrt/c:pjrt_c_api_gpu_plugin.so` for all other accelerators.

The PJRT plugin can then be found in `bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so` or `bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so` depending on the accelerator.

## Metal
Apple has developed a closed source Metal PJRT plugin. To obtain it, download the [latest release of `jax-metal` from Pypi](https://pypi.org/project/jax-metal/#files). Then decompress the wheel and the plugin should be located in `jax_plugins/metal_plugin/pjrt_plugin_metal_14.dylib`.

The Metal plugin has many bugs and missing features. To enable workarounds, call `Metal.enable ()` in the main framework.

## TPU
The TPU PJRT plugin is closed-sourced can be downloaded on the [TPU runtimes webpage](https://cloud.google.com/tpu/docs/runtimes).
