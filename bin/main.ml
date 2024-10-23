open Iree_bindings

(* let () = *)
(*   Compile.compile *)
(* {| *)
   (*    func.func @abs(%a : tensor<f32>, %b: tensor<f32>) -> (tensor<i1>) { *)
   (*      %result = "stablehlo.compare"(%a, %b) { *)
   (*     comparison_direction = #stablehlo<comparison_direction EQ> *)
   (* } : (tensor<f32>, tensor<f32>) -> tensor<i1> *)
   (*      "func.return"(%result) : (tensor<i1>) -> () *)
   (*    } *)
   (*    |} *)
(*     "out.vmfb" *)

(* let example_code = *)
(*   Stable_hlo. *)
(*     { id= "abs" *)
(*     ; inputs= [("a", Tensor_type ([], F32)); ("b", Tensor_type ([], F32))] *)
(*     ; outputs= [Tensor_type ([], I1)] *)
(*     ; body= *)
(*         [ { inputs= [("a", Tensor_type ([], F32)); ("b", Tensor_type ([], F32))] *)
(*           ; outputs= [("result", Tensor_type ([], I1))] *)
(*           ; name= "stablehlo.compare" *)
(*           ; attributes= *)
(*               [("comparison_direction", "#stablehlo<comparison_direction EQ>")] *)
(*           } *)
(*         ; { inputs= [("result", Tensor_type ([], I1))] *)
(*           ; outputs= [] *)
(*           ; name= "func.return" *)
(*           ; attributes= [] } ] } *)

(* let stable_hlo = Stable_hlo.func_to_string example_code *)

(* let () = *)
(*   print_endline stable_hlo ; *)
(*   Compile.compile stable_hlo "out.vmfb" *)

let example_code =
  Ir.(
    let* a = Argument [] in
    let* b = Argument [] in
    let* result1 = Compare (a, Le, b) in
    let* result2 = Absf a in
    vars_to_func [result1; result2] )
  |> Stable_hlo.func_to_string

let () =
  print_endline example_code ;
  Compile.compile example_code "out.vmfb"
