open Iree_bindings

let main =
  Ir.(
    create_func
      [Tensor_type ([1], F32); Tensor_type ([1], F32)]
      (fun a b ->
        let* result1 = Compare (a, Le, b) in
        let* result2 = Absf a in
        VarList.[result1; result2] ) )

let main_code = Ir.func_to_stable_hlo main |> Stable_hlo.func_to_string

let () =
  print_endline main_code ;
  Compile.compile main_code "out.vmfb"
