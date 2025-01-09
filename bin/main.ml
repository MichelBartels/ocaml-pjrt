open Pjrt_bindings

let example_code =
  {|
func.func @main(%arg0: tensor<4xf32>, %arg1:
tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
|}

let client =
  Client.make
    "/home/michel/part-ii-project/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"

let executable = Client.compile client example_code

let device = Client.devices client |> List.hd

let x = Client.buffer_to_device client device [1.0; 2.0; 3.0; 4.0] [4]

let y = Client.buffer_to_device client device [1.1; 1.2; 1.3; 1.4] [4]

let output = Client.execute client 1 executable [x; y] |> List.hd

let output = Client.buffer_to_host client 4 output

let () =
  Printf.printf "Output: %s\n"
    (String.concat ", " (List.map string_of_float output))
