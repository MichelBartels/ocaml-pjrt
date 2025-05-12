module Device =
  ( val Pjrt_bindings.make ~caching:false
          "/Users/michelbartels/Downloads/pjrt/jax_plugins/metal_plugin/pjrt_plugin_metal_14.dylib"
    )

open Device

let example_code =
  {|
   func.func @main(%arg0: tensor<4xf32>, %arg1:
   tensor<4xf32>) -> tensor<4xf32> {
     %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
     "func.return"(%0) : (tensor<4xf32>) -> ()
   }
       |}

let program = compile ~path:"cached.bin" example_code

let x = Device_api.Tensor.of_list F32 [4] [1.0; 2.0; 3.0; 4.0]

let x = tensor_to_buffer x

let y = Device_api.Tensor.of_list F32 [4] [1.1; 1.2; 1.3; 1.4]

let y = tensor_to_buffer y

let outputs = execute program ~num_outputs:1 [x; y]

let output = List.hd outputs

let output = buffer_to_tensor F32 ~shape:[4] output

let output = Device_api.Tensor.to_list output

let () =
  Printf.printf "Output: %s\n"
    (String.concat ", " (List.map string_of_float output))
