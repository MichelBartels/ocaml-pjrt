open Iree_bindings

let () =
  Compile.compile
    {|
   func.func @abs(%input : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
     %result = "math.absf"(%input) : (tensor<f32>) -> tensor<f32>
     "func.return"(%result, %input) : (tensor<f32>, tensor<f32>) -> ()
   }
   |}
    "out.vmfb"

(* let example_code = *)
(*   Stable_hlo. *)
(*     { id= "abs" *)
(*     ; inputs= [("input", Tensor_type ([], F32))] *)
(*     ; outputs= [Tensor_type ([], F32)] *)
(*     ; body= *)
(*         [ { inputs= [("input", Tensor_type ([], F32))] *)
(*           ; outputs= [("result", Tensor_type ([], F32))] *)
(*           ; name= "math.absf" } *)
(*         ; { inputs= [("result", Tensor_type ([], F32))] *)
(*           ; outputs= [] *)
(*           ; name= "func.return" } ] } *)

(* let stable_hlo = Stable_hlo.func_to_string example_code *)

(* let () = *)
(*   print_endline stable_hlo ; *)
(*   Compile.compile stable_hlo "out.vmfb" *)

let example_code =
  Ir.(
    let* input = Argument () in
    let* result = Absf input in
    vars_to_func [result] )
  |> Stable_hlo.func_to_string

let () =
  print_endline example_code ;
  Compile.compile example_code "out.vmfb"
