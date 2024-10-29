open Iree_bindings

let main =
  Dsl.(
    let add =
      fn
        [Tensor_type ([], F32); Tensor_type ([], F32)]
        (fun x y -> [x + y; x + x])
    in
    fn
      [Tensor_type ([], F32)]
      (fun x ->
        let [x; y] = Ir.call_func add [x; x] in
        [x + y] ) )

let main_code = Ir.compile main

let () =
  print_endline main_code ;
  Compile.compile main_code "out.vmfb"
